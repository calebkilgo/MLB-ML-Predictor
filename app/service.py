from __future__ import annotations

from app.schemas import PredictRequest, PredictResponse
from src.betting.edge import edge, expected_value
from src.models.predict import GameInput, predict


def run_prediction(req: PredictRequest) -> PredictResponse:
    g = GameInput(
        home_team=req.home_team.upper(),
        away_team=req.away_team.upper(),
        home_sp_era=req.home_sp_era, away_sp_era=req.away_sp_era,
        home_sp_fip=req.home_sp_fip, away_sp_fip=req.away_sp_fip,
        home_sp_k9=req.home_sp_k9,   away_sp_k9=req.away_sp_k9,
        home_sp_bb9=req.home_sp_bb9, away_sp_bb9=req.away_sp_bb9,
        temp_c=req.temp_c, wind_kph=req.wind_kph, precip_mm=req.precip_mm,
        home_rest=req.home_rest, away_rest=req.away_rest,
    )
    out = predict(g)
    resp = PredictResponse(**out)
    if req.home_ml_odds is not None:
        resp.home_edge = edge(out["p_home_win"], req.home_ml_odds)
        resp.home_ev = expected_value(out["p_home_win"], req.home_ml_odds)
    if req.away_ml_odds is not None:
        resp.away_edge = edge(out["p_away_win"], req.away_ml_odds)
        resp.away_ev = expected_value(out["p_away_win"], req.away_ml_odds)
    return resp
