from __future__ import annotations

import threading
import time
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, Form, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app import auth, daily_context, mlb_live, prediction_log
from app.schemas import PredictRequest, PredictResponse
from app.service import run_prediction
from src.adjustments import apply_adjustments, total_runs_adjustment
from src.models.predict import GameInput, predict

BASE = Path(__file__).parent
app = FastAPI(title="MLB Predict", version="0.9.0")
app.mount("/static", StaticFiles(directory=BASE / "static"), name="static")
templates = Jinja2Templates(directory=BASE / "templates")

# A simple admin session cookie, separate from the user session cookie.
ADMIN_COOKIE = "mlbp_admin"
ADMIN_SESSION_SECS = 4 * 3600  # 4 hours

REFRESH_SECS = 300
_BOARD: dict = {"ready": False, "games": [], "built_at": 0, "building": False}
_LOCK = threading.Lock()


# ---------------------------------------------------------------------------
# Board builder (unchanged from v8)
# ---------------------------------------------------------------------------

def _process_game(g: dict) -> dict:
    try:
        gi = GameInput(
            home_team=g["home_team"],
            away_team=g["away_team"],
            home_pitcher_id=g.get("home_pitcher_id"),
            away_pitcher_id=g.get("away_pitcher_id"),
        )
        base = predict(gi)
        ctx = daily_context.build_context(g)
        p_adj, p_break = apply_adjustments(base["p_home_win"], ctx)
        t_adj, t_break = total_runs_adjustment(base["proj_total_runs"], ctx)

        weight = 0.45 + 0.1 * p_adj
        home_runs = t_adj * weight
        away_runs = t_adj - home_runs

        g["model"] = {
            "p_home_win": p_adj,
            "p_away_win": 1 - p_adj,
            "proj_total_runs": t_adj,
            "proj_home_runs": home_runs,
            "proj_away_runs": away_runs,
            "confidence": min(1.0, abs(p_adj - 0.5) * 2 + 0.1),
        }
        g["adjustments"] = {
            "probability": p_break,
            "total_runs": t_break,
            "context": {k: v for k, v in ctx.items()
                        if k not in ("home_team", "away_team")},
        }
    except Exception as e:
        g["model"] = {"error": str(e)}
        g["adjustments"] = {"error": str(e)}
    return g


def _build_board() -> None:
    with _LOCK:
        if _BOARD["building"]:
            return
        _BOARD["building"] = True
    try:
        t0 = time.time()
        start, end = mlb_live.get_default_window()
        print(f"[board] building {start} -> {end}")
        games = mlb_live.get_schedule(start, end)
        print(f"[board] {len(games)} games, enriching...")

        with ThreadPoolExecutor(max_workers=16) as ex:
            enriched = list(ex.map(_process_game, games))

        try:
            log_stats = prediction_log.record_games(enriched)
            print(f"[log] new={log_stats['new']} "
                  f"resolved={log_stats['resolved']} "
                  f"total={log_stats['total']}")
        except Exception as e:
            print(f"[log] failed: {e}")

        with _LOCK:
            _BOARD["games"] = enriched
            _BOARD["window_start"] = start.isoformat()
            _BOARD["window_end"] = end.isoformat()
            _BOARD["built_at"] = time.time()
            _BOARD["ready"] = True
        print(f"[board] done in {time.time() - t0:.1f}s")
    except Exception as e:
        print(f"[board] build failed: {e}")
    finally:
        with _LOCK:
            _BOARD["building"] = False


def _refresh_loop() -> None:
    while True:
        time.sleep(REFRESH_SECS)
        _build_board()


@app.on_event("startup")
def _startup() -> None:
    threading.Thread(target=_build_board, daemon=True).start()
    threading.Thread(target=_refresh_loop, daemon=True).start()


# ---------------------------------------------------------------------------
# Auth helpers
# ---------------------------------------------------------------------------

def _user_authed(request: Request) -> bool:
    cookie = request.cookies.get(auth.COOKIE_NAME)
    return auth.parse_session_cookie(cookie) is not None


def _admin_authed(request: Request) -> bool:
    cookie = request.cookies.get(ADMIN_COOKIE)
    if not cookie:
        return False
    rec = auth.parse_session_cookie(cookie)
    # Reuse the session-cookie machinery but check an "admin" sentinel kid.
    # The admin cookie is actually a different payload; simpler: check the
    # raw signature ourselves.
    return _verify_admin_cookie(cookie)


def _make_admin_cookie() -> tuple[str, int]:
    import hmac, hashlib, json, base64
    secret = auth._get_secret()
    payload = json.dumps(
        {"admin": True, "exp": int(time.time()) + ADMIN_SESSION_SECS},
        separators=(",", ":"),
    ).encode()
    sig = hmac.new(secret, payload, hashlib.sha256).digest()
    p = base64.urlsafe_b64encode(payload).rstrip(b"=").decode()
    s = base64.urlsafe_b64encode(sig).rstrip(b"=").decode()
    return f"{p}.{s}", ADMIN_SESSION_SECS


def _verify_admin_cookie(value: str | None) -> bool:
    import hmac, hashlib, json, base64
    if not value or "." not in value:
        return False
    try:
        p_b64, s_b64 = value.split(".", 1)
        pad = "=" * (-len(p_b64) % 4)
        payload = base64.urlsafe_b64decode(p_b64 + pad)
        pad = "=" * (-len(s_b64) % 4)
        sig = base64.urlsafe_b64decode(s_b64 + pad)
    except Exception:
        return False
    expected = hmac.new(auth._get_secret(), payload, hashlib.sha256).digest()
    if not hmac.compare_digest(sig, expected):
        return False
    try:
        data = json.loads(payload.decode())
        if not data.get("admin"):
            return False
        if int(data.get("exp", 0)) <= int(time.time()):
            return False
    except Exception:
        return False
    return True


def _gate(request: Request) -> Response | None:
    """Return a redirect/401 response if the request isn't authed, else None."""
    if _user_authed(request):
        return None
    if request.url.path.startswith("/api/"):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    return RedirectResponse(url="/unlock", status_code=302)


# ---------------------------------------------------------------------------
# Public (unauthed) routes
# ---------------------------------------------------------------------------

@app.get("/unlock", response_class=HTMLResponse)
def unlock_get(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(request, "unlock.html", {"error": None})


@app.post("/unlock", response_class=HTMLResponse)
def unlock_post(request: Request, key: str = Form(...)) -> Response:
    ip = request.client.host if request.client else None
    rec = auth.validate_and_touch(key.strip(), ip)
    if not rec:
        return templates.TemplateResponse(
            request, "unlock.html",
            {"error": "Invalid or expired key."},
            status_code=401,
        )
    value, max_age = auth.make_session_cookie(rec)
    resp = RedirectResponse(url="/", status_code=302)
    resp.set_cookie(
        auth.COOKIE_NAME, value, max_age=max_age,
        httponly=True, samesite="lax", path="/",
    )
    return resp


@app.get("/logout")
def logout() -> Response:
    resp = RedirectResponse(url="/unlock", status_code=302)
    resp.delete_cookie(auth.COOKIE_NAME, path="/")
    return resp


# ---------- admin ----------

def _render_admin(request: Request, *, authed: bool,
                  error: str | None = None,
                  new_key=None) -> HTMLResponse:
    keys_view = []
    if authed:
        for k in auth.list_keys():
            now = int(time.time())
            expired = k.expires_at <= now
            if not k.active:
                status, status_class = "REVOKED", "revoked"
            elif expired:
                status, status_class = "EXPIRED", "expired"
            else:
                status, status_class = "ACTIVE", "active"
            keys_view.append({
                "label": k.label, "kid": k.kid, "key": k.key,
                "duration": auth.DURATION_LABELS.get(k.duration, k.duration),
                "expires_human": datetime.fromtimestamp(k.expires_at)
                                .strftime("%Y-%m-%d %H:%M"),
                "use_count": k.use_count,
                "status": status, "status_class": status_class,
                "active": k.active, "expired": expired,
            })
        keys_view.sort(key=lambda x: (x["status"] != "ACTIVE", x["label"]))

    new_key_view = None
    if new_key is not None:
        new_key_view = {
            "key": new_key.key,
            "label": new_key.label,
            "duration": auth.DURATION_LABELS.get(new_key.duration, new_key.duration),
            "expires_human": datetime.fromtimestamp(new_key.expires_at)
                             .strftime("%Y-%m-%d %H:%M"),
        }
    return templates.TemplateResponse(
        request, "admin.html",
        {
            "authed": authed,
            "error": error,
            "keys": keys_view,
            "new_key": new_key_view,
        },
    )


@app.get("/admin", response_class=HTMLResponse)
def admin_get(request: Request) -> HTMLResponse:
    return _render_admin(request, authed=_admin_authed(request))


@app.post("/admin/login", response_class=HTMLResponse)
def admin_login(request: Request, password: str = Form(...)) -> Response:
    if not auth.admin_password_ok(password):
        return _render_admin(request, authed=False,
                             error="Wrong password.")
    value, max_age = _make_admin_cookie()
    resp = RedirectResponse(url="/admin", status_code=302)
    resp.set_cookie(ADMIN_COOKIE, value, max_age=max_age,
                    httponly=True, samesite="lax", path="/")
    return resp


@app.post("/admin/logout")
def admin_logout() -> Response:
    resp = RedirectResponse(url="/admin", status_code=302)
    resp.delete_cookie(ADMIN_COOKIE, path="/")
    return resp


@app.post("/admin/create", response_class=HTMLResponse)
def admin_create(request: Request,
                 label: str = Form(...),
                 duration: str = Form(...)) -> Response:
    if not _admin_authed(request):
        return RedirectResponse(url="/admin", status_code=302)
    try:
        rec = auth.create_key(label=label, duration=duration)
    except ValueError as e:
        return _render_admin(request, authed=True, error=str(e))
    return _render_admin(request, authed=True, new_key=rec)


@app.post("/admin/revoke", response_class=HTMLResponse)
def admin_revoke(request: Request, kid: str = Form(...)) -> Response:
    if not _admin_authed(request):
        return RedirectResponse(url="/admin", status_code=302)
    auth.revoke_key(kid)
    return RedirectResponse(url="/admin", status_code=302)


# ---------------------------------------------------------------------------
# Gated routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
def index(request: Request) -> Response:
    gate = _gate(request)
    if gate is not None:
        return gate
    return templates.TemplateResponse(request, "index.html", {})


@app.get("/api/games")
def api_games(request: Request) -> Response:
    gate = _gate(request)
    if gate is not None:
        return gate
    with _LOCK:
        snap = dict(_BOARD)
    if not snap.get("ready"):
        return JSONResponse({
            "warming": True,
            "games": [],
            "message": "Board warming up — first build takes ~30s.",
        })
    return JSONResponse({
        "warming": False,
        "window_start": snap.get("window_start"),
        "window_end": snap.get("window_end"),
        "built_at": snap.get("built_at"),
        "age_seconds": int(time.time() - snap.get("built_at", 0)),
        "games": snap.get("games", []),
    })


@app.get("/api/calibration")
def api_calibration(request: Request) -> Response:
    gate = _gate(request)
    if gate is not None:
        return gate
    return JSONResponse(prediction_log.summary())


@app.post("/api/predict", response_model=PredictResponse)
def api_predict(request: Request, req: PredictRequest) -> Response:
    gate = _gate(request)
    if gate is not None:
        return gate
    return JSONResponse(run_prediction(req).model_dump())


@app.get("/api/health")
def health() -> dict:
    # health is unauthed so you can monitor it without a cookie
    with _LOCK:
        return {"ok": True, "board_ready": _BOARD.get("ready", False),
                "built_at": _BOARD.get("built_at", 0)}