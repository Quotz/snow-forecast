# Snow Forecast — Popova Shapka

Automated powder forecast system for Popova Shapka ski resort (North Macedonia). Fetches weather data from 4 sources (Open-Meteo, Yr.no, Snow-Forecast, Mountain-Forecast), calculates composite powder scores (0-100), detects weather patterns, and generates a static HTML dashboard. Runs 3x daily via GitHub Actions and publishes to GitHub Pages.

## Live Dashboard

**[View the forecast →](https://YOUR_USERNAME.github.io/snow-forecast/)**

## Setup

1. **Fork** this repository
2. **Enable GitHub Pages**: Settings → Pages → Source: `main` branch, folder: `/docs`
3. **Add Telegram secrets** (optional): Settings → Secrets → Actions
   - `TELEGRAM_BOT_TOKEN` — get from [@BotFather](https://t.me/BotFather)
   - `TELEGRAM_CHAT_ID` — your chat or group ID
4. **Trigger a run**: Actions → Snow Forecast → Run workflow

The forecast will then run automatically at 06:00, 12:00, and 18:00 CET.

## Customizing for a Different Resort

Edit `config.yaml`:
- Update `locations` with your resort's coordinates and elevations
- Update `scrapers` with matching Snow-Forecast / Mountain-Forecast keys
- Adjust `scoring` thresholds to match your local conditions

## Local Development

```bash
pip install -r requirements.txt
python main.py --no-notify
open docs/index.html
```

### CLI Options

| Flag | Description |
|------|-------------|
| `--no-notify` | Skip Telegram notifications |
| `--dashboard-only` | Regenerate dashboard from existing data |
| `--json-only` | Output JSON only, no HTML |
| `--verbose` / `-v` | Debug logging |

## Project Structure

| File | Purpose |
|------|---------|
| `main.py` | Orchestration — fetch, score, report, notify |
| `data_extract.py` | Transform raw API responses into scoring data |
| `analysis.py` | Chart data, safety flags, source comparison |
| `scoring.py` | Powder score calculation (0-100) |
| `patterns.py` | Weather pattern detection |
| `report.py` | HTML dashboard generation (Jinja2) |
| `notify.py` | Telegram notifications |
| `collectors/` | Data source adapters (Open-Meteo, Yr.no, scrapers) |
| `config.yaml` | All configuration and thresholds |
