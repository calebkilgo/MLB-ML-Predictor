"""Access-key gate for the MLB Predict app.

Storage layout (all in data/access/):
  keys.json   — list of issued keys with metadata
  secret.txt  — 32-byte signing secret for session cookies (auto-generated)

Session cookies are signed with HMAC-SHA256 over JSON
    {"kid": "<key_id>", "exp": <unix_ts>}
The cookie value is base64(payload) + "." + base64(hmac).

Keys are random 24-byte url-safe strings. Admin password comes from the
ADMIN_PASSWORD environment variable; if unset, admin access is disabled.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from threading import Lock
from typing import Any

from src.config import CFG

_ACCESS_DIR = CFG.data_dir / "access"
_KEYS_PATH = _ACCESS_DIR / "keys.json"
_SECRET_PATH = _ACCESS_DIR / "secret.txt"
_LOCK = Lock()

COOKIE_NAME = "mlbp_session"

DURATIONS: dict[str, int] = {
    "1d": 1 * 24 * 3600,
    "3d": 3 * 24 * 3600,
    "1w": 7 * 24 * 3600,
    "1mo": 30 * 24 * 3600,
}
DURATION_LABELS: dict[str, str] = {
    "1d": "1 day",
    "3d": "3 days",
    "1w": "1 week",
    "1mo": "1 month",
}


@dataclass
class KeyRecord:
    kid: str               # short public identifier
    key: str               # the actual secret string users paste in
    label: str             # human-readable label ("Tyler", "Dad")
    created_at: int        # unix seconds
    expires_at: int        # unix seconds
    duration: str          # "1d" | "3d" | "1w" | "1mo"
    active: bool = True
    use_count: int = 0
    last_used_at: int | None = None
    last_used_ip: str | None = None


# ---------- storage helpers ----------

def _ensure_dir() -> None:
    _ACCESS_DIR.mkdir(parents=True, exist_ok=True)


def _get_secret() -> bytes:
    _ensure_dir()
    if not _SECRET_PATH.exists():
        _SECRET_PATH.write_bytes(secrets.token_bytes(32))
    return _SECRET_PATH.read_bytes()


def _load_keys() -> list[KeyRecord]:
    _ensure_dir()
    if not _KEYS_PATH.exists():
        return []
    try:
        data = json.loads(_KEYS_PATH.read_text())
    except Exception:
        return []
    return [KeyRecord(**row) for row in data]


def _save_keys(keys: list[KeyRecord]) -> None:
    _ensure_dir()
    _KEYS_PATH.write_text(json.dumps([asdict(k) for k in keys], indent=2))


# ---------- public API ----------

def create_key(label: str, duration: str) -> KeyRecord:
    if duration not in DURATIONS:
        raise ValueError(f"duration must be one of {list(DURATIONS)}")
    with _LOCK:
        keys = _load_keys()
        kid = secrets.token_urlsafe(6)
        key = secrets.token_urlsafe(18)
        now = int(time.time())
        rec = KeyRecord(
            kid=kid, key=key, label=label or "(no label)",
            created_at=now,
            expires_at=now + DURATIONS[duration],
            duration=duration,
        )
        keys.append(rec)
        _save_keys(keys)
        return rec


def revoke_key(kid: str) -> bool:
    with _LOCK:
        keys = _load_keys()
        for k in keys:
            if k.kid == kid:
                k.active = False
                _save_keys(keys)
                return True
        return False


def list_keys() -> list[KeyRecord]:
    return _load_keys()


def _find_by_key(raw_key: str) -> KeyRecord | None:
    for k in _load_keys():
        if hmac.compare_digest(k.key, raw_key):
            return k
    return None


def validate_and_touch(raw_key: str, ip: str | None) -> KeyRecord | None:
    """Check a user-entered key. On success, record usage and return it."""
    k = _find_by_key(raw_key)
    if not k or not k.active:
        return None
    now = int(time.time())
    if k.expires_at <= now:
        return None
    with _LOCK:
        keys = _load_keys()
        for kk in keys:
            if kk.kid == k.kid:
                kk.use_count += 1
                kk.last_used_at = now
                kk.last_used_ip = ip
                _save_keys(keys)
                return kk
    return None


def validate_kid(kid: str) -> KeyRecord | None:
    """Look up a key by its kid. Returns None if unknown, inactive, or expired."""
    now = int(time.time())
    for k in _load_keys():
        if k.kid == kid and k.active and k.expires_at > now:
            return k
    return None


# ---------- session cookies ----------

def _b64e(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).rstrip(b"=").decode("ascii")


def _b64d(s: str) -> bytes:
    pad = "=" * (-len(s) % 4)
    return base64.urlsafe_b64decode(s + pad)


def make_session_cookie(rec: KeyRecord) -> tuple[str, int]:
    """Return (cookie_value, max_age_seconds)."""
    payload = json.dumps({"kid": rec.kid, "exp": rec.expires_at},
                         separators=(",", ":")).encode()
    sig = hmac.new(_get_secret(), payload, hashlib.sha256).digest()
    value = f"{_b64e(payload)}.{_b64e(sig)}"
    max_age = max(60, rec.expires_at - int(time.time()))
    return value, max_age


def parse_session_cookie(value: str | None) -> KeyRecord | None:
    if not value or "." not in value:
        return None
    try:
        p_b64, s_b64 = value.split(".", 1)
        payload = _b64d(p_b64)
        sig = _b64d(s_b64)
    except Exception:
        return None
    expected = hmac.new(_get_secret(), payload, hashlib.sha256).digest()
    if not hmac.compare_digest(sig, expected):
        return None
    try:
        data = json.loads(payload.decode())
        kid = data.get("kid")
        exp = int(data.get("exp", 0))
    except Exception:
        return None
    if exp <= int(time.time()):
        return None
    return validate_kid(kid)


# ---------- admin password ----------

def admin_password_ok(provided: str) -> bool:
    expected = os.getenv("ADMIN_PASSWORD", "").strip()
    if not expected:
        return False
    return hmac.compare_digest(provided.strip(), expected)