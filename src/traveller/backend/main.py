"""
FastAPI server exposing the TravelAgent to the React frontend.

Run with:
    uvicorn main:app --reload --port 8000

Then open frontend/index.html in a browser (see README for details).
"""

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from agent import TravelAgent
from sight_agent import get_curated_sight

app = FastAPI(title="Travel Agent API")

# Allow the frontend to call this API even if it's ever opened from a different origin.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

agent = TravelAgent()


class TripRequest(BaseModel):
    origin_id: str
    destination_id: str


@app.get("/api/destinations")
def get_destinations():
    return [
        {"id": d.id, "name": d.name, "lat": d.lat, "lon": d.lon}
        for d in agent.list_destinations()
    ]


@app.post("/api/trip")
def post_trip(req: TripRequest):
    try:
        return agent.plan_trip(req.origin_id, req.destination_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e))


@app.get("/api/sights/{destination_id}")
def get_sights(destination_id: str):
    try:
        return agent.get_top_sights(destination_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e))


@app.get("/api/sight/{xid}")
def get_sight_detail(xid: str, name: str = ""):
    try:
        return get_curated_sight(xid, fallback_name=name)
    except RuntimeError as e:
        raise HTTPException(status_code=502, detail=str(e))


@app.get("/api/health")
def health():
    return {"status": "ok"}


# Serve the frontend (index.html + assets) from the same server, at the root
# path. This must be added AFTER the /api routes above, so those take
# priority - Starlette matches routes in the order they were registered.
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
