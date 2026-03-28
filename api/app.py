"""FastAPI backend for Snow Forecast — serves forecast data, radar, and community features."""
from __future__ import annotations

import json
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

PROJECT_DIR = Path(__file__).parent.parent
DOCS_DIR = PROJECT_DIR / "docs"

app = FastAPI(
    title="Snow Forecast API",
    description="Powder forecast API for Popova Shapka",
    version="4.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Serve the static dashboard
if DOCS_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(DOCS_DIR)), name="static")


@app.get("/")
async def root():
    """Serve the dashboard HTML."""
    index = DOCS_DIR / "index.html"
    if index.exists():
        return FileResponse(str(index))
    return {"message": "Snow Forecast API", "docs": "/docs"}


@app.get("/api/forecast")
async def get_forecast():
    """Return the latest forecast data."""
    latest = DOCS_DIR / "latest.json"
    if not latest.exists():
        raise HTTPException(404, "No forecast data available")
    with open(latest) as f:
        return json.load(f)


@app.get("/api/forecast/scores")
async def get_scores():
    """Return just the score cards (lighter payload)."""
    latest = DOCS_DIR / "latest.json"
    if not latest.exists():
        raise HTTPException(404, "No forecast data available")
    with open(latest) as f:
        data = json.load(f)
    return {
        "location": data.get("location"),
        "generated_at": data.get("generated_at"),
        "go_verdict": data.get("go_verdict"),
        "scores": data.get("scores", [])[:7],
        "insights": data.get("insights"),
    }


@app.get("/api/forecast/extended")
async def get_extended():
    """Return extended 16-day scores."""
    latest = DOCS_DIR / "latest.json"
    if not latest.exists():
        raise HTTPException(404, "No forecast data available")
    with open(latest) as f:
        data = json.load(f)
    return {
        "scores": data.get("scores", []),
        "seasonal_outlook": data.get("seasonal_outlook"),
        "seasonal_context": data.get("seasonal_context"),
    }


@app.get("/api/radar")
async def get_radar():
    """Return current radar frame URLs."""
    latest = DOCS_DIR / "latest.json"
    if not latest.exists():
        raise HTTPException(404, "No forecast data available")
    with open(latest) as f:
        data = json.load(f)
    radar = data.get("radar")
    if not radar:
        return {"frames": [], "message": "No radar data available"}
    return radar


@app.get("/api/verification")
async def get_verification():
    """Return verification statistics."""
    stats_path = DOCS_DIR / "verification" / "stats.json"
    if not stats_path.exists():
        return {"message": "No verification data yet"}
    with open(stats_path) as f:
        return json.load(f)


@app.get("/api/health")
async def health():
    """Health check endpoint."""
    latest = DOCS_DIR / "latest.json"
    has_data = latest.exists()
    generated_at = None
    if has_data:
        try:
            with open(latest) as f:
                data = json.load(f)
            generated_at = data.get("generated_at")
        except Exception:
            pass
    return {
        "status": "ok" if has_data else "no_data",
        "latest_forecast": generated_at,
    }
