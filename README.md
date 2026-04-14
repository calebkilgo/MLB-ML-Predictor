# MLB Predict

Calibrated ML for MLB win-probability and run-total forecasting, with a
FastAPI + vanilla-JS UI. Research and educational use only.

## Quickstart

```bash
git clone <repo> mlb-predict && cd mlb-predict
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env          # edit seasons if desired

make etl        # pulls games, team/pitcher stats, weather (slow first run)
make features   # builds data/processed/features.parquet
make train      # fits calibrated classifier + run regressor, saves to models/
make backtest   # walk-forward metrics + reliability plots in reports/
make app        # http://localhost:8000
make test
```

## How it works

* **ETL** uses `pybaseball` for Baseball-Reference/FanGraphs data and
  Open-Meteo for historical weather. All raw tables land in `data/raw/`.
* **Features** include ELO with home advantage, 10-game rolling runs scored/
  allowed and win%, rest days, prior-season team wRC+/FIP, starter ERA/FIP/K9/
  BB9, park factor, and game-day weather. All rolling features use
  `shift(1)` to guarantee no target leakage.
* **Model** is LightGBM → `CalibratedClassifierCV(method="isotonic")`, trained
  on train seasons and calibrated on a held-out validation season. Primary
  metrics: Brier score, log loss. Secondary: ROC AUC. A reliability diagram
  is written to `reports/` on every run.
* **Run totals** use a LightGBM Tweedie regressor on the same features.
* **Betting helpers** (`src/betting/edge.py`) convert model probabilities into
  EV, edge, and capped Kelly fractions against American odds.

## Extending

* Swap `src/etl/retrosheet.py` for direct Retrosheet GL parsing if you want
  pre-2000 data.
* Add bullpen FIP, lineup wOBA vs LHP/RHP, travel miles, and umpire factors
  as new columns in `src/features/assemble.py` — they'll flow to the model
  automatically via `FEATURE_COLS`.
* Replace the Jinja UI with a React app hitting `/api/predict` — the API is
  stable and typed via Pydantic.

## Disclaimer

This project is for research and informational use only. It does not
constitute betting advice, and no model guarantees profit. Gamble responsibly.
