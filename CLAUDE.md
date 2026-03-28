# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

Automated powder forecast system for Popova Shapka ski resort (North Macedonia). Fetches weather data from 7+ sources, runs 7 forecast models (5 physics + 2 AI), calculates composite powder scores (0-100), detects weather patterns, and generates an interactive HTML dashboard. Runs 3x daily via GitHub Actions, deploys to GitHub Pages, and sends Telegram alerts when conditions are noteworthy.

Live dashboard: https://quotz.github.io/snow-forecast/

## Commands

```bash
# Local development (use venv)
source .venv/bin/activate  # or: python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python main.py --no-notify              # Full run without Telegram alerts
python main.py --no-notify --verify     # Also run ERA5 verification
python main.py --dashboard-only         # Regenerate HTML from existing latest.json
python main.py --json-only              # Output JSON only, no HTML
python main.py --verbose                # Debug logging

# Open generated dashboard
open docs/index.html

# Backtesting (local only, not CI)
python backtest.py --days 30
python backtest.py --start-date 2025-12-01 --end-date 2026-01-31
```

No unit tests. Validation via ERA5 verification system and manual inspection.

## Architecture

### Pipeline Flow

`main.py` orchestrates: **Fetch → Extract → Validate → Score → Patterns → Insights/Verdict → Report → Notify → Verify → Recalibrate**

### Data Sources (collectors/)

All collectors inherit from `BaseCollector` with `fetch()` / `safe_fetch()` methods.

- `open_meteo.py` — Fetches ALL 7 models (5 physics + 2 AI) in a single `/v1/forecast` request, including pressure-level data (700hPa temp, 500hPa GPH). Primary data source.
- `open_meteo_ensemble.py` — Probabilistic ensemble (ICON-EU EPS, ECMWF IFS ENS, GFS ENS) for uncertainty bands
- `yr_no.py` — MET Norway API, independent validation source
- `snow_forecast.py` / `mountain_forecast.py` — HTML scrapers (share `scraper_base.py`)
- `seasonal.py` — Open-Meteo EC46 seasonal forecast (46-day outlook, weekly anomalies)
- `rainviewer.py` — RainViewer radar tile index (free, real-time precipitation radar)
- `meteoblue.py` — Paid API, disabled by default (needs `METEOBLUE_API_KEY`)

### Scoring & Analysis

- `data_extract.py` — Transforms raw hourly API responses into per-day scoring data. Includes DGZ detection, bluebird classification, and crystal type estimation.
- `scoring.py` — Powder score (0-100) with 10 weighted components. Supports Kalman filter bias correction (`kalman.py`) and lead-time-dependent adaptive model weighting.
- `snow_physics.py` — Dendritic Growth Zone (DGZ) detection, surface hoar risk assessment, bluebird day classification from 500hPa ridging, crystal type estimation.
- `patterns.py` — Detects 9 weather patterns (storm-then-clear with bluebird classifier, multi-day storm, warming trend, cold snap, upslope, wind-slab, melt-freeze, surface hoar risk, standalone bluebird day).
- `ensemble_stats.py` — Model correlation/clustering, CRPS, Brier score, analog storage for ML.
- `analysis.py` — Chart.js data, safety flags, avalanche danger, model spread, spaghetti plot data, probability fan data.
- `validation.py` — Cross-source sanity checks (temperature outliers, snowfall contradictions)
- `insights.py` — Dashboard insights, "Should I Go?" verdict (`compute_go_verdict()`), seasonal context formatting.

### Output & Notifications

- `report.py` — Jinja2 rendering with `templates/dashboard.html`, outputs to `docs/`
- `notify.py` — Smart Telegram alerts (GOOD/EPIC days, snow watches, pattern alerts). Deduplication via `docs/.last_alert`
- `subscribers.py` — Telegram `/start`, `/stop`, `/report` commands. Persists to `.subscribers.json`
- `crowd_reports.py` — Parses and stores crowd-sourced condition reports from Telegram
- `forecast_diff.py` — Detects significant forecast changes between runs
- `history.py` — Aggregates last 90 forecast files for trend charts

### Verification & Self-Improvement

- `verification.py` — Compares forecasts against ERA5 reanalysis. Computes per-model MAE/RMSE, CRPS, Brier scores. Updates Kalman filter with each verification. Stores analog pairs for future ML.
- `kalman.py` — Univariate Kalman filter per model per variable for adaptive bias correction. State persists to `docs/verification/kalman_state.json`.
- `recalibration.py` — Weekly auto-recalibration (Sundays): recomputes weights, adjusts Kalman noise parameters, detects seasonal regime changes.
- `backtest.py` — Standalone historical validation (physics models only)

## Key Technical Details

- **All configuration in `config.yaml`** — thresholds, models, locations, alert rules, Kalman parameters, webcam URLs. No secrets (env vars for Telegram).
- **AI model IDs**: `ecmwf_aifs025_single` (AIFS) and `gfs_graphcast025` (GraphCast). All fetched via Open-Meteo `/v1/forecast`, NOT `/v1/ecmwf`.
- **Pressure-level data**: 700hPa temperature (DGZ), 500hPa geopotential height (bluebird ridging) requested alongside standard params.
- **Output directory is `docs/`** for GitHub Pages. Contains `latest.json` (32 keys), `index.html`, `history/`, `manifest.json`, `.source_health.json`, `.nojekyll`.
- **Python 3.12** in CI, local dev may be 3.9. Use `from __future__ import annotations` for 3.9 compat.
- **No async, no ORM** — pure synchronous Python with requests/jinja2/pyyaml/beautifulsoup4.
- **VPS**: `forecast.815431624.xyz` at `46.225.98.16` (Hetzner, Caddy reverse proxy, Docker). For Phase 5 API deployment.
- **CI workflow** (`forecast.yml`): Runs at 05:00, 11:00, 17:00 UTC. Commits `docs/` and `.subscribers.json`.
- **Repo**: https://github.com/Quotz/snow-forecast (GitHub user: Quotz)

## Overhaul Plan

See `OVERHAUL_PLAN.md` for the full v4.0 implementation plan with completion checklist.
