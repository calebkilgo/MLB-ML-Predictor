from __future__ import annotations

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    home_team: str = Field(..., min_length=2, max_length=4)
    away_team: str = Field(..., min_length=2, max_length=4)
    home_sp_era: float = 4.0
    away_sp_era: float = 4.0
    home_sp_fip: float = 4.0
    away_sp_fip: float = 4.0
    home_sp_k9: float = 8.5
    away_sp_k9: float = 8.5
    home_sp_bb9: float = 3.0
    away_sp_bb9: float = 3.0
    temp_c: float = 22.0
    wind_kph: float = 12.0
    precip_mm: float = 0.0
    home_rest: float = 1.0
    away_rest: float = 1.0
    home_ml_odds: int | None = None
    away_ml_odds: int | None = None


class PredictResponse(BaseModel):
    p_home_win: float
    p_away_win: float
    proj_total_runs: float
    proj_home_runs: float
    proj_away_runs: float
    confidence: float
    home_edge: float | None = None
    away_edge: float | None = None
    home_ev: float | None = None
    away_ev: float | None = None
