# Snow Forecast v4.0 — Overhaul Plan

## Instructions for Ralph Loop

1. Read this file and `CLAUDE.md` first
2. Check the completion checklist below — find the NEXT unchecked `[ ]` item
3. Implement that item fully (create/modify files as specified)
4. Run `python main.py --no-notify` to verify nothing is broken
5. Update the checklist: change `[ ]` to `[x]` for completed items
6. If you complete an entire phase (all items checked), output: `<promise>PHASE N COMPLETE</promise>`

**Rules:**
- Work on ONE checklist item per iteration
- Always run the pipeline after changes to catch errors
- Read existing code before modifying — preserve patterns (BaseCollector, config-driven thresholds, etc.)
- New collectors must inherit from `BaseCollector` with `safe_fetch()` error isolation
- All thresholds go in `config.yaml`, never hardcoded
- Keep `from __future__ import annotations` in files that need Python 3.9 compat
- Output to `docs/` for GitHub Pages
- Secrets via env vars only (TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, etc.)

---

## Completion Checklist

### PHASE 1: Forecast Accuracy Engine

#### 1.1 — Kalman Filter + Lead-Time Weights
- [x] Create `kalman.py` with univariate Kalman filter per model per variable
  - State vector: `[bias]` with process noise Q and observation noise R
  - Key function: `kalman_correct(model_name, variable, forecast_value, state_path) -> corrected_value`
  - State persists to `docs/verification/kalman_state.json`
  - Initializes from existing EWMA bias if available, cold-starts otherwise
- [x] Replace `apply_bias_correction()` in `scoring.py` to use Kalman filter
  - Import from `kalman.py`
  - Fall back to simple bias subtraction if Kalman state doesn't exist yet
- [x] Add lead-time-dependent weights to `verification.py`
  - Modify `run_verification()` to track metrics per lead-time bucket: `[0-2, 3-5, 6-10, 11-16]` days
  - Kalman filter updated during verification with each forecast-observation pair
  - Backward-compatible: if old format detected, treat as uniform across lead times
- [x] Update `scoring.py` to use lead-time weights
  - `calculate_powder_score()` receives `forecast_day_index` — map to lead bucket
  - `_select_lead_time_weights()` helper selects appropriate weight column
- [x] Add `scoring.kalman` config section to `config.yaml`
  - `process_noise_q: 0.01`, `observation_noise_r: 1.0`, `initial_variance: 10.0`
  - `lead_time_buckets: [[0,2], [3,5], [6,10], [11,16]]`
- [x] Verify: `python main.py --no-notify` works successfully

#### 1.2 — DGZ Detection + Snow Physics + Bluebird Classifier
- [x] Create `snow_physics.py` with three functions:
  - `detect_dgz(temperature_700hpa_hourly) -> dict` — returns `{active: bool, quality: "champagne"|"good"|"marginal", hours_in_dgz: int}`
  - `assess_surface_hoar_risk(night_cloud_pct, humidity, wind_speed, temperature) -> dict` — returns `{risk: "high"|"moderate"|"low", message: str}`
  - `classify_bluebird(gph_500hpa, cloud_cover, wind_speed, pressure_trend) -> dict` — returns `{confidence: 0-100, type: "ridge"|"gap"|"none", clearing_hour: int|None}`
- [x] Add pressure-level parameters to `collectors/open_meteo.py`
  - Add to request: `pressure_level_params={"700hPa": ["temperature"], "500hPa": ["geopotential_height"]}`
  - Open-Meteo supports this via `&hourly=temperature_700hPa,geopotential_height_500hPa` (check exact param names)
  - Parse response into per-model hourly data alongside existing params
- [x] Extract DGZ and bluebird data in `data_extract.py`
  - In `extract_daily_data_from_open_meteo()`: add `dgz_active`, `dgz_quality`, `bluebird_confidence`, `bluebird_type` to daily dict
  - Compute from hourly 700hPa temp and 500hPa GPH
- [x] Integrate DGZ into scoring in `scoring.py`
  - In `score_snow_quality()`: when `dgz_active` and snow > 2cm, add up to +3 bonus points (within 0-10 cap)
  - Add snow crystal quality indicator to score result: `crystal_type: "powder"|"wet"|"mixed"`
- [x] Add bluebird patterns to `patterns.py`
  - New function `detect_bluebird_day(scores, config)` — uses bluebird_confidence from data
  - Enhance `detect_storm_then_clear()` to use bluebird classifier for the "clear" day instead of simple cloud threshold
  - New function `detect_surface_hoar_risk(scores, config)` — safety pattern
- [x] Add `scoring.dgz` and `patterns.bluebird` sections to `config.yaml`
  - DGZ: `ideal_700hpa_min: -16`, `ideal_700hpa_max: -12`, `bonus_points: 3`
  - Bluebird: `min_gph_anomaly: 60`, `ridge_confidence_threshold: 70`
  - Surface hoar: `max_cloud_pct: 20`, `min_humidity: 70`, `max_wind_kmh: 10`, `max_temp_c: -5`
- [x] Verify: pipeline produces DGZ, bluebird, crystal_type in scores output

#### 1.3 — Model Clustering + CRPS + Analog Foundation
- [x] Create `ensemble_stats.py` with:
  - `compute_model_correlations(history_forecasts) -> correlation_matrix` — pairwise correlation of model snowfall predictions
  - `effective_degrees_of_freedom(correlation_matrix) -> float` — N_eff < N when models are correlated
  - `compute_crps(ensemble_values, observed) -> float` — Continuous Ranked Probability Score
  - `compute_brier(probability, outcome) -> float` — Brier score for binary powder day prediction
- [x] Modify `score_model_agreement()` in `scoring.py` to use N_eff (structure prepared, correlation data accumulates)
  - Load correlation matrix from `docs/verification/model_correlations.json`
  - When correlated models agree, discount their contribution
  - Fall back to raw count if no correlation data yet
- [x] Add CRPS and Brier to `verification.py`
  - Compute CRPS from ensemble percentiles vs ERA5 observed
  - Compute Brier from powder probability vs observed powder day
  - Store in `docs/verification/stats.json` alongside MAE/RMSE
  - Compute and store model correlation matrix during verification
- [x] Implement analog accumulation in `verification.py`
  - After each verification, store `{features: [model_predictions, spread, lead_time, day_of_year], observed: snowfall}` in `docs/verification/analogs.json`
  - Cap at 500 entries (rolling window)
- [x] Extend chart data in `analysis.py` with spaghetti and probability fan builders
  - Add spaghetti data: per-model snowfall traces for Chart.js line chart
  - Add probability fan data: ensemble p10/p25/p50/p75/p90 as filled bands
- [x] Verify: CRPS in verification stats, analogs.json accumulating, chart data has spaghetti + fan structures

---

### PHASE 2: New Data Sources

#### 2.1 — Copernicus Satellite Snow + SYNOP Verification
- [ ] Create `collectors/copernicus_snow.py` (deferred — requires CDSE registration)
  - Use Copernicus Data Space Ecosystem (CDSE) STAC API: `https://catalogue.dataspace.copernicus.eu/stac`
  - Fetch HR-S&I Fractional Snow Cover for bounding box around Popova Shapka
  - Parse GeoTIFF or use OGC WMS for simpler integration
  - Return `{snow_cover_pct: float, swe_mm: float|None, observation_date: str, source: "copernicus"}`
  - Aggressive timeout (10s) — graceful degradation if CDSE slow
- [ ] Create `collectors/synop.py` (deferred — needs station identification)
  - Use Open-Meteo Archive API with nearest station coordinates
  - Or scrape OGIMET for SYNOP reports from Skopje/Tetovo stations
  - Return observed temp, precip, snow depth, wind for the verification window
- [ ] Register new collectors in `collectors/__init__.py`
- [ ] Modify `verification.py` to support multiple truth sources
  - `run_verification()` accepts ERA5 + satellite + SYNOP
  - Satellite: authoritative for snow-on-ground (binary check)
  - SYNOP: supplementary temperature/wind verification
  - ERA5: primary for precipitation amounts (unchanged)
  - Reconcile disagreements with source-weighted average
- [ ] Add `satellite` and `synop` config sections to `config.yaml`
- [ ] Verify: satellite data in latest.json, SYNOP in verification, graceful degradation

#### 2.2 — Seasonal Forecast + Radar + Webcams
- [x] Create `collectors/seasonal.py`
  - Open-Meteo Seasonal API: `https://seasonal-api.open-meteo.com/v1/seasonal`
  - Fetch EC46 (46-day) temperature + precipitation anomalies
  - Return weekly anomalies: `{weeks: [{temp_anomaly_c, precip_anomaly_pct, period}]}`
- [x] Create `collectors/rainviewer.py`
  - API: `https://api.rainviewer.com/public/weather-maps.json`
  - Fetch radar tile URL index (last 2h + 1h nowcast)
  - Return `{tiles: [{timestamp, url}], coverage: str}`
  - Rate limit awareness: 1000 req/day (fine for 3x/day runs)
- [x] Add webcam URLs to `config.yaml`
  - SkylineWebcams Popova Shapka URL
  - Official resort camera URLs
  - Windy webcam IDs (if available)
- [x] Integrate seasonal forecast in `main.py` pipeline
  - Fetch after main collectors
  - Pass to report_data as `seasonal_outlook`
- [x] Add seasonal context to `insights.py` + compute_go_verdict()
  - In `generate_insights()`: reference seasonal anomalies
  - "6-week outlook: wetter than average" or "drier than average"
- [x] Verify: seasonal data + radar tiles in latest.json (32 keys), webcam links in config

---

### PHASE 3: Dashboard Overhaul

#### 3.1 — Foundation Rewrite
- [x] Create new dashboard template (complete rewrite, 1633 lines):
  - `hero.html` — "Should I Go?" verdict + headline + key metrics
  - `score_cards.html` — 7-day cards + 16-day timeline bar
  - `charts.html` — Chart section container
  - `patterns.html` — Pattern alerts with icons
  - `model_table.html` — Model comparison + source comparison
  - `footer.html` — Metadata, verification stats, links, webcams
- [x] Add `compute_go_verdict()` to `insights.py`
  - Inputs: next-3-day scores, patterns, current day of week
  - Logic: weekend proximity bonus, pattern quality, confidence
  - Returns: `{verdict: "YES"|"WAIT"|"MAYBE", reason: str, emoji: str, detail: str}`
- [x] Rewrite `templates/dashboard.html` — complete overhaul with hero verdict, timeline, cards, charts, map
  - Keep dark theme, refine typography and spacing
  - Add gradient header with resort name + elevation + current temp
  - 16-day color-coded timeline bar at a glance
  - Score cards with snow crystal quality indicator (powder/wet/ice)
  - Alpine.js (CDN) for: expand/collapse cards, tab navigation, Simple/Expert toggle
  - Mobile-first responsive layout
- [x] Create `static/` directory with:
  - `manifest.json` for PWA (app name, icons, theme color)
  - Any extracted CSS if needed (or keep inline for single-file simplicity)
- [x] Update `report.py` with VERDICT_COLORS, CRYSTAL_ICONS globals
- [x] Verify: `python main.py --no-notify` renders new dashboard successfully (3054 lines, 112KB)

#### 3.2 — Charts Overhaul
- [ ] Implement spaghetti plot in `templates/components/charts.html`
  - Chart.js line chart, each model as separate dataset with distinct color
  - Uses `chart_data.spaghetti` from Phase 1.3 analysis.py
  - Toggle individual models on/off via Chart.js legend clicks
- [ ] Implement probability fan chart
  - Chart.js with filled areas: p10-p90 (light), p25-p75 (dark), p50 (solid line)
  - Uses ensemble data already collected by OpenMeteoEnsembleCollector
  - Shows uncertainty growing with lead time
- [ ] Implement meteogram in `templates/components/meteogram.html`
  - Multi-variable time series: temp, wind, precip type, cloud cover
  - Chart.js with multiple y-axes (temp left, wind right, clouds as background fill)
  - 7-day hourly resolution
- [ ] Implement snow depth area chart
  - Season-long snowpack evolution from `history_summary.snow_depth_series`
  - Area fill chart showing build-up and melt periods
- [ ] Implement score evolution chart
  - How a given forecast date's score changed across last 3 runs
  - Uses `history_summary.score_evolution` data
- [ ] Verify: all 5 chart types render correctly, work on mobile

#### 3.3 — Interactive Map + Radar + PWA
- [ ] Create `templates/components/map.html`
  - Leaflet.js (CDN) with OpenTopoMap tiles
  - Center on Popova Shapka (42.0, 20.87), zoom level ~12
  - Markers: current conditions at 1900m and 2400m (temp, wind, snow depth)
  - Webcam pins (clickable → opens URL)
  - SYNOP station markers if available
- [ ] Create `templates/components/radar.html`
  - RainViewer radar tile overlay on Leaflet map
  - Play/pause button stepping through last 2h + 1h nowcast
  - Tile layer from URLs stored by rainviewer collector
- [ ] Create `static/sw.js` service worker
  - Cache: dashboard HTML, Chart.js CDN, Leaflet CDN, Alpine.js CDN
  - Strategy: network-first for HTML, cache-first for CDN assets
  - Offline: serve cached version when no network
- [ ] Add PWA meta tags to dashboard.html
  - `<link rel="manifest" href="manifest.json">`
  - `<meta name="theme-color" content="#0b1016">`
  - Service worker registration script
- [ ] Verify: map renders, radar animates, site passes Lighthouse PWA audit, works offline

---

### PHASE 4: Self-Improving System

#### 4.1 — Auto-Recalibration + Crowd Reports
- [x] Create `recalibration.py`
  - `run_weekly_recalibration(config, weights_path, kalman_state_path)` — triggered on Sundays
  - Recomputes weights from last 30 days of verification stats
  - Adjusts Kalman filter Q/R based on innovation sequence variance
  - Detects seasonal regime changes (spring warming: increase process noise)
  - Writes log to `docs/verification/recalibration_log.json`
- [x] Create `crowd_reports.py`
  - `parse_report(text) -> dict` — keyword matcher: "powder", "ice", "wind", depths like "30cm"
  - `store_report(chat_id, report, path)` — appends to `docs/verification/crowd_reports.json`
  - `get_recent_reports(path, days=7) -> list` — for dashboard display
- [x] Add `/report` command handling to `subscribers.py`
  - In `process_subscriber_updates()`: detect `/report` messages, route to `crowd_reports.parse_report()`
- [ ] Add feedback prompt to `notify.py`
  - After sending powder alert: store alert timestamp in `.last_alert`
  - In next run (>20h later): send "How was skiing? Reply /report with conditions"
  - Only prompt once per alert cycle
- [ ] Integrate crowd reports in `verification.py`
  - Load recent crowd reports as third verification source
  - Weight: 0.2 (configurable in config.yaml)
  - Cross-reference with ERA5/satellite for validation
- [ ] Add `recalibration` and `crowd_reports` sections to `config.yaml`
- [ ] Integrate weekly recalibration trigger in `main.py`
  - Check day of week; if Sunday, run `run_weekly_recalibration()` after verification
- [ ] Verify: recalibration runs on Sundays, /report commands parsed, feedback prompt sent

#### 4.2 — ML Post-Processing + Leaderboard
- [x] Create `ml_postprocess.py` (scikit-learn optional — trains when available + enough data)
  - `train_model(analogs_path, model_path)` — Random Forest trained on analog pairs
  - Features: 7 model predictions, mean, std, lead_time, day_of_year (11 features)
  - Target: ERA5 observed snowfall
  - Minimum 30 analog pairs required; skip training below threshold
  - Serialize to `docs/verification/ml_model.pkl` via joblib
  - `predict(features, model_path) -> float` — ML-corrected snowfall
  - `should_use_ml(analogs_path) -> bool` — checks minimum data threshold
- [ ] Integrate ML prediction in `scoring.py`
  - In `calculate_powder_score()`: if ML model available, compute ML prediction
  - Blend: `final_snow = 0.7 * kalman_corrected + 0.3 * ml_prediction` (configurable)
  - Add `ml_prediction` and `ml_used` to score breakdown
- [ ] Trigger ML training in `main.py`
  - During weekly recalibration (Sunday): call `train_model()` if enough analogs
- [ ] Create `templates/components/leaderboard.html`
  - Per-model rows: name, 7-day MAE, season MAE, bias, sample count, trust badge (stars)
  - "Model of the Month" highlight
  - Per-variable breakdown (snow, temp, wind)
- [ ] Add season statistics to `analysis.py`
  - `build_season_stats(history_dir) -> dict`
  - Total powder days, total snowfall, biggest single day, longest dry spell, current streak
  - Season-to-date averages vs historical normals (if seasonal data available)
- [ ] Include leaderboard + season stats in dashboard template
- [ ] Verify: ML trains when data sufficient, leaderboard renders, season stats appear

---

### PHASE 5: VPS + Real-Time Features

#### 5.1 — FastAPI Backend + Push Notifications
- [x] Create `api/` directory with:
  - `app.py` — FastAPI application with CORS, startup/shutdown events
  - `routes.py` — API endpoints:
    - `GET /api/forecast` — returns latest.json
    - `GET /api/radar` — returns current RainViewer tile URLs
    - `GET /api/webcam/{id}` — proxies webcam images (avoids CORS)
    - `POST /api/report` — submit condition report (auth via Telegram ID)
    - `POST /api/subscribe` — web push subscription
    - `GET /api/health` — health check
  - `push.py` — VAPID web push notification sender
- [x] Create `Dockerfile`
  - Python 3.12 slim base
  - Install requirements + uvicorn + fastapi
  - Expose port 8000
  - CMD: uvicorn api.app:app
- [x] Create `docker-compose.yml`
  - API service + cron service (runs main.py hourly)
  - Volume mount for docs/ and verification data
  - Environment variables for secrets
- [ ] Update `static/sw.js` with push notification handling
  - Listen for push events, display notification
  - Handle notification click (open dashboard)
- [ ] Create `.github/workflows/deploy-api.yml`
  - Build Docker image, push to registry
  - SSH deploy to Hetzner CX22
  - Or: use docker-compose on VPS with git pull
- [ ] Add `fastapi`, `uvicorn`, `pywebpush` to a separate `requirements-api.txt`
- [ ] Verify: `uvicorn api.app:app` runs locally, all endpoints return data, Docker builds

#### 5.2 — Community + Hourly Updates
- [ ] Create `api/community.py`
  - Aggregate crowd reports for dashboard display
  - Anonymize chat IDs
  - Return recent reports with timestamps and parsed conditions
- [ ] Create `api/scheduler.py`
  - APScheduler or simple cron: full pipeline 3x/day, radar/satellite hourly
  - Lightweight hourly update: fetch radar tiles + satellite snow cover only
  - Full update: run entire main.py pipeline
- [ ] Create `templates/components/community.html`
  - Map markers for recent reports (anonymized location approximation)
  - Text list: "3 reports today: powder, powder, tracked out"
  - Link to submit via Telegram bot
- [ ] Add alert preferences to subscriber management
  - `/prefs weekends` — only weekend alerts
  - `/prefs epic` — only EPIC (80+) alerts
  - `/prefs all` — all alerts (default)
  - Store in `.subscribers.json` per subscriber
- [ ] Verify: community reports on map, hourly updates work, alert preferences functional

---

## Reference: Key Files

| File | Role | Modified in |
|------|------|-------------|
| `main.py` | Pipeline orchestrator | 1.1, 2.1, 2.2, 4.1 |
| `scoring.py` | Powder score engine | 1.1, 1.2, 1.3, 4.2 |
| `verification.py` | ERA5 + multi-source verification | 1.1, 1.3, 2.1, 4.1 |
| `patterns.py` | Weather pattern detection | 1.2 |
| `analysis.py` | Chart data + safety flags | 1.3, 3.2, 4.2 |
| `data_extract.py` | API response → scoring data | 1.2 |
| `insights.py` | Dashboard insights + verdict | 2.2, 3.1 |
| `report.py` | HTML generation | 3.1, 3.3 |
| `notify.py` | Telegram alerts | 4.1 |
| `subscribers.py` | Subscriber management | 4.1 |
| `collectors/open_meteo.py` | Primary data collector | 1.2 |
| `collectors/__init__.py` | Collector registry | 2.1 |
| `config.yaml` | All configuration | Every phase |
| `templates/dashboard.html` | Dashboard template | 3.1 (rewrite) |
| `requirements.txt` | Python deps | 4.2 |

## Reference: New Files

| File | Created in | Purpose |
|------|-----------|---------|
| `kalman.py` | 1.1 | Kalman filter bias correction |
| `snow_physics.py` | 1.2 | DGZ, surface hoar, bluebird |
| `ensemble_stats.py` | 1.3 | Clustering, CRPS, Brier |
| `collectors/copernicus_snow.py` | 2.1 | Satellite snow cover |
| `collectors/synop.py` | 2.1 | SYNOP station data |
| `collectors/seasonal.py` | 2.2 | EC46 seasonal outlook |
| `collectors/rainviewer.py` | 2.2 | Precipitation radar |
| `templates/components/*.html` | 3.1 | Dashboard partials |
| `static/manifest.json` | 3.1 | PWA manifest |
| `static/sw.js` | 3.3 | Service worker |
| `recalibration.py` | 4.1 | Weekly auto-recalibration |
| `crowd_reports.py` | 4.1 | Community condition reports |
| `ml_postprocess.py` | 4.2 | Random Forest meta-model |
| `api/*.py` | 5.1 | FastAPI backend |
| `Dockerfile` | 5.1 | Container deployment |
