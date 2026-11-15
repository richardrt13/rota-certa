import asyncio
import json
import os
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from typing import List, Tuple
from zoneinfo import ZoneInfo
import asyncpg
import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from shapely.geometry import LineString, Point, shape

load_dotenv()

DATABASE_URL = os.environ["DATABASE_URL"]
ORS_API_KEY = os.environ["ORS_API_KEY"]
ADMIN_PIN = os.environ.get("ADMIN_PIN", "")
ORS_DIRECTIONS_URL = "https://api.openrouteservice.org/v2/directions/driving-car/geojson"
RIO_TZ = ZoneInfo("America/Sao_Paulo")

db_pool: asyncpg.Pool | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global db_pool
    db_pool = await asyncpg.create_pool(DATABASE_URL, min_size=1, max_size=5)
    yield
    await db_pool.close()


app = FastAPI(title="Rota Certa RJ - V1", lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")


class RouteRequest(BaseModel):
    origin: Tuple[float, float]       # (lat, lng)
    destination: Tuple[float, float]  # (lat, lng)


class CustomAvoidZoneRequest(BaseModel):
    name: str = Field(..., min_length=2, max_length=120)
    lat: float
    lng: float
    radius_meters: int = Field(default=350, ge=100, le=2000)


def verify_admin_pin(x_admin_pin: str | None):
    if ADMIN_PIN and x_admin_pin != ADMIN_PIN:
        raise HTTPException(status_code=401, detail="PIN administrativo inválido.")


def get_carjacking_time_multiplier() -> float:
    hour = datetime.now(RIO_TZ).hour
    if 21 <= hour or hour < 5:
        return 2.5   # Madrugada / noite
    if 18 <= hour < 21 or 5 <= hour < 7:
        return 1.6   # Entardecer / início da manhã
    return 1.0


async def fetch_corridor_avoid_zones(
    origin: Tuple[float, float], destination: Tuple[float, float]
) -> List[dict]:
    min_lng = min(origin[1], destination[1]) - 0.12
    min_lat = min(origin[0], destination[0]) - 0.12
    max_lng = max(origin[1], destination[1]) + 0.12
    max_lat = max(origin[0], destination[0]) + 0.12

    async with db_pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT id, name, category, severity, ST_AsGeoJSON(geom) AS geojson
            FROM avoid_zones
            WHERE (expires_at IS NULL OR expires_at > NOW())
              AND geom && ST_MakeEnvelope($1, $2, $3, $4, 4326)
            LIMIT 45;
            """,
            min_lng, min_lat, max_lng, max_lat,
        )
    return [
        {
            "id": r["id"],
            "name": r["name"],
            "category": r["category"],
            "severity": r["severity"],
            "geometry": json.loads(r["geojson"]),
        }
        for r in rows
    ]


async def evaluate_carjacking_exposure(route_coords: List[List[float]]) -> dict:
    wkt_line = "LINESTRING(" + ", ".join(f"{lng} {lat}" for lng, lat in route_coords) + ")"
    time_mult = get_carjacking_time_multiplier()

    async with db_pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT
                precinct_name,
                roubo_veiculo_6m,
                risk_index,
                ST_Length(
                    ST_Intersection(ST_GeomFromText($1, 4326)::geography, geom::geography)
                ) AS meters_in_cisp
            FROM precinct_risk
            WHERE ST_Intersects(ST_GeomFromText($1, 4326), geom);
            """,
            wkt_line,
        )

    total_score = 0.0
    warnings = []
    for r in rows:
        km_exposed = r["meters_in_cisp"] / 1000.0
        risk_idx = float(r["risk_index"])
        total_score += km_exposed * (risk_idx / 10.0) * time_mult
        if risk_idx >= 55.0 and km_exposed >= 0.5:
            warnings.append(
                f"{r['precinct_name']} ({r['roubo_veiculo_6m']} roubos/6m, {km_exposed:.1f} km)"
            )

    return {
        "carjacking_score": round(min(total_score, 100.0), 1),
        "time_multiplier": time_mult,
        "high_risk_precincts": warnings,
    }


async def call_ors(
    client: httpx.AsyncClient,
    origin: Tuple[float, float],
    destination: Tuple[float, float],
    avoid_multipolygon: dict | None = None,
    request_alternatives: bool = False,
) -> dict:
    headers = {"Authorization": ORS_API_KEY, "Content-Type": "application/json"}
    body: dict = {
        "coordinates": [[origin[1], origin[0]], [destination[1], destination[0]]],
        "radiuses": [1500, 1500],
        "instructions": True,
    }
    if avoid_multipolygon:
        body["options"] = {"avoid_polygons": avoid_multipolygon}
    elif request_alternatives:
        body["alternative_routes"] = {
            "target_count": 3,
            "weight_factor": 1.6,
            "share_factor": 0.6,
        }

    resp = await client.post(ORS_DIRECTIONS_URL, json=body, headers=headers)
    if resp.status_code != 200:
        raise HTTPException(status_code=502, detail=f"Erro OpenRouteService: {resp.text}")
    return resp.json()


def extract_arterial_step_waypoint(
    baseline_coords: List[List[float]],
    chosen_feature: dict,
) -> List[Tuple[float, float]]:
    safe_coords = chosen_feature["geometry"]["coordinates"]
    baseline_line = LineString(baseline_coords)

    step_indices = set()
    for seg in chosen_feature.get("properties", {}).get("segments", []):
        for step in seg.get("steps", []):
            wp_range = step.get("way_points", [])
            if wp_range:
                step_indices.add(wp_range[0])

    min_idx = max(1, len(safe_coords) // 8)
    max_idx = len(safe_coords) - min_idx
    candidate_indices = [i for i in step_indices if min_idx <= i <= max_idx] or list(range(min_idx, max_idx))

    best_point = None
    max_dist = 0.0
    for idx in candidate_indices:
        lng, lat = safe_coords[idx]
        dist = baseline_line.distance(Point(lng, lat))
        if dist > max_dist:
            max_dist = dist
            best_point = (round(lat, 6), round(lng, 6))

    if best_point and max_dist > 0.0028:
        return [best_point]
    return []


def build_google_maps_url(
    origin: Tuple[float, float],
    destination: Tuple[float, float],
    waypoints: List[Tuple[float, float]],
) -> str:
    url = (
        f"https://www.google.com/maps/dir/?api=1"
        f"&origin={origin[0]},{origin[1]}"
        f"&destination={destination[0]},{destination[1]}"
        f"&travelmode=driving"
    )
    if waypoints:
        wp_param = "|".join(f"{lat},{lng}" for lat, lng in waypoints)
        url += f"&waypoints={wp_param}"
    return url


@app.get("/")
async def serve_ui():
    return FileResponse("static/index.html")


@app.get("/health")
async def health_check():
    async with db_pool.acquire() as conn:
        await conn.fetchval("SELECT 1;")
    return {"status": "ok"}


@app.get("/api/map-layers")
async def get_map_layers():
    async with db_pool.acquire() as conn:
        avoid_rows = await conn.fetch(
            """
            SELECT id, name, category, severity, ST_AsGeoJSON(geom) AS geojson
            FROM avoid_zones
            WHERE expires_at IS NULL OR expires_at > NOW();
            """
        )
        precinct_rows = await conn.fetch(
            """
            SELECT cisp_id, precinct_name, roubo_veiculo_6m, risk_index, ST_AsGeoJSON(geom) AS geojson
            FROM precinct_risk
            WHERE risk_index >= 50.0;
            """
        )

    return {
        "avoid_zones": {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "properties": {
                        "id": r["id"],
                        "name": r["name"],
                        "category": r["category"],
                        "severity": r["severity"],
                    },
                    "geometry": json.loads(r["geojson"]),
                }
                for r in avoid_rows
            ],
        },
        "high_carjacking_precincts": {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "properties": {
                        "precinct_name": r["precinct_name"],
                        "roubo_veiculo_6m": r["roubo_veiculo_6m"],
                        "risk_index": float(r["risk_index"]),
                    },
                    "geometry": json.loads(r["geojson"]),
                }
                for r in precinct_rows
            ],
        },
    }


@app.post("/api/avoid-zones")
async def create_custom_avoid_zone(
    zone: CustomAvoidZoneRequest,
    x_admin_pin: str | None = Header(default=None),
):
    verify_admin_pin(x_admin_pin)
    ext_id = f"manual_{uuid.uuid4().hex[:10]}"
    async with db_pool.acquire() as conn:
        await conn.execute(
            """
            INSERT INTO avoid_zones (external_id, name, category, severity, expires_at, geom)
            VALUES (
                $1, $2, 'carjacking_hotspot', 90, NULL,
                ST_SimplifyPreserveTopology(
                    ST_Buffer(ST_SetSRID(ST_MakePoint($3, $4), 4326)::geography, $5)::geometry,
                    0.00015
                )
            );
            """,
            ext_id,
            zone.name,
            zone.lng,
            zone.lat,
            zone.radius_meters,
        )
    return {"status": "created", "external_id": ext_id}


@app.delete("/api/avoid-zones/{zone_id}")
async def delete_avoid_zone(
    zone_id: int,
    x_admin_pin: str | None = Header(default=None),
):
    verify_admin_pin(x_admin_pin)
    async with db_pool.acquire() as conn:
        await conn.execute("DELETE FROM avoid_zones WHERE id = $1;", zone_id)
    return {"status": "deleted", "id": zone_id}


@app.post("/api/safe-route")
async def compute_safe_route(req: RouteRequest):
    zones = await fetch_corridor_avoid_zones(req.origin, req.destination)
    zone_shapes = [(z, shape(z["geometry"])) for z in zones]

    avoid_mp = None
    if zones:
        avoid_mp = {
            "type": "MultiPolygon",
            "coordinates": [z["geometry"]["coordinates"] for z in zones],
        }

    async with httpx.AsyncClient(timeout=20.0) as client:
        tasks = [call_ors(client, req.origin, req.destination, None, request_alternatives=True)]
        if avoid_mp:
            tasks.append(call_ors(client, req.origin, req.destination, avoid_mp, request_alternatives=False))
        results = await asyncio.gather(*tasks)

    alt_features = results[0].get("features", [])
    baseline_feat = alt_features[0]
    baseline_coords = baseline_feat["geometry"]["coordinates"]

    candidate_features = list(alt_features)
    if len(results) > 1:
        candidate_features.extend(results[1].get("features", []))

    evaluated = []
    for idx, feat in enumerate(candidate_features):
        coords = feat["geometry"]["coordinates"]
        line = LineString(coords)
        hits = [z["name"] for z, poly in zone_shapes if line.intersects(poly)]
        cj = await evaluate_carjacking_exposure(coords)
        dur_min = feat["properties"]["summary"]["duration"] / 60.0

        hard_penalty = 10000.0 if hits else 0.0
        combined_cost = hard_penalty + dur_min + (cj["carjacking_score"] * 0.35)

        evaluated.append({
            "index": idx,
            "feature": feat,
            "coords": coords,
            "hits": hits,
            "carjacking": cj,
            "duration_min": dur_min,
            "combined_cost": combined_cost,
        })

    best = min(evaluated, key=lambda x: x["combined_cost"])
    base_eval = evaluated[0]

    waypoints = []
    if best["index"] != 0:
        waypoints = extract_arterial_step_waypoint(baseline_coords, best["feature"])

    if base_eval["hits"] and not best["hits"]:
        reason = f"⚠️ Desvio de bloqueio ({', '.join(set(base_eval['hits']))}) | +{round(best['duration_min'] - base_eval['duration_min'], 1)} min"
    elif best["index"] != 0:
        risk_diff = round(base_eval["carjacking"]["carjacking_score"] - best["carjacking"]["carjacking_score"], 1)
        time_diff = round(best["duration_min"] - base_eval['duration_min'], 1)
        reason = f"🛡️ Rota alternativa mais segura (-{risk_diff} pts de risco, +{time_diff} min)"
    else:
        reason = "✅ Rota padrão já é a opção mais segura disponível"

    summary = best["feature"]["properties"]["summary"]
    return {
        "detour_applied": best["index"] != 0,
        "selection_reason": reason,
        "avoided_zones": base_eval["hits"],
        "duration_minutes": round(summary["duration"] / 60, 1),
        "distance_km": round(summary["distance"] / 1000, 2),
        "carjacking": best["carjacking"],
        "injected_waypoints": waypoints,
        "baseline_geojson": baseline_feat,
        "safe_geojson": best["feature"],
        "google_maps_url": build_google_maps_url(req.origin, req.destination, waypoints),
    }
