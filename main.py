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
from shapely.ops import unary_union

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


app = FastAPI(title="Rota Certa RJ - V1.1", lifespan=lifespan)
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
        return 2.5
    if 18 <= hour < 21 or 5 <= hour < 7:
        return 1.6
    return 1.0


async def fetch_zones_near_corridors(candidate_lines: List[LineString]) -> List[dict]:
    """
    Busca apenas zonas de bloqueio num raio de 3 km das rotas candidatas.
    Evita estourar o limite de 200 km² de avoid_polygons do OpenRouteService.
    """
    corridor_geom = unary_union(candidate_lines)
    async with db_pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT
                id, name, category, severity,
                ST_AsGeoJSON(geom) AS geojson,
                ST_Distance(geom::geography, ST_GeomFromText($1, 4326)::geography) AS dist_m
            FROM avoid_zones
            WHERE (expires_at IS NULL OR expires_at > NOW())
              AND ST_DWithin(geom::geography, ST_GeomFromText($1, 4326)::geography, 3000)
            ORDER BY dist_m ASC
            LIMIT 60;
            """,
            corridor_geom.wkt,
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


def extract_corridor_waypoints(
    baseline_coords: List[List[float]],
    chosen_feature: dict,
    max_waypoints: int = 4,
) -> List[Tuple[float, float]]:
    """
    Extrai até 4 pontos sobre manobras (steps) distribuídos ao longo da rota segura
    para impedir que o Google Maps volte a cortar por dentro de comunidades.
    """
    safe_coords = chosen_feature["geometry"]["coordinates"]
    n_coords = len(safe_coords)
    if n_coords < 10:
        return []

    baseline_line = LineString(baseline_coords)

    step_indices = sorted({
        step["way_points"][0]
        for seg in chosen_feature.get("properties", {}).get("segments", [])
        for step in seg.get("steps", [])
        if step.get("way_points")
    })
    if not step_indices:
        step_indices = list(range(n_coords))

    # Divide o trajeto em 4 fatias (15% a 85%) e escolhe 1 ponto de desvio por fatia
    waypoints: List[Tuple[float, float]] = []
    slice_bounds = [(0.15, 0.32), (0.33, 0.50), (0.51, 0.68), (0.69, 0.85)]

    for start_pct, end_pct in slice_bounds:
        if len(waypoints) >= max_waypoints:
            break
        i_min = int(n_coords * start_pct)
        i_max = int(n_coords * end_pct)
        bucket = [i for i in step_indices if i_min <= i <= i_max]
        if not bucket:
            continue

        best_idx = max(
            bucket,
            key=lambda idx: baseline_line.distance(Point(safe_coords[idx][0], safe_coords[idx][1])),
        )
        lng, lat = safe_coords[best_idx]
        # Só adiciona se houver desvio real (> ~200m) em relação à rota padrão
        if baseline_line.distance(Point(lng, lat)) > 0.0018:
            waypoints.append((round(lat, 6), round(lng, 6)))

    return waypoints


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
    async with httpx.AsyncClient(timeout=20.0) as client:
        # 1. Busca alternativas livres primeiro para mapear os corredores possíveis
        alt_res = await call_ors(client, req.origin, req.destination, None, request_alternatives=True)
        alt_features = alt_res.get("features", [])
        if not alt_features:
            raise HTTPException(status_code=404, detail="Nenhuma rota encontrada.")

        baseline_feat = alt_features[0]
        baseline_coords = baseline_feat["geometry"]["coordinates"]
        candidate_lines = [LineString(f["geometry"]["coordinates"]) for f in alt_features]

        # 2. Busca apenas zonas de bloqueio num raio de 3 km desses corredores
        zones = await fetch_zones_near_corridors(candidate_lines)
        zone_shapes = [(z, shape(z["geometry"])) for z in zones]

        candidate_features = list(alt_features)

        # 3. Se alguma alternativa invadir uma comunidade/bloqueio, pede rota com avoid_polygons
        any_hit = any(
            line.intersects(poly)
            for line in candidate_lines
            for _, poly in zone_shapes
        )
        if zones and any_hit:
            avoid_mp = {
                "type": "MultiPolygon",
                "coordinates": [z["geometry"]["coordinates"] for z in zones],
            }
            safe_res = await call_ors(client, req.origin, req.destination, avoid_mp, request_alternatives=False)
            candidate_features.extend(safe_res.get("features", []))

    evaluated = []
    for idx, feat in enumerate(candidate_features):
        coords = feat["geometry"]["coordinates"]
        line = LineString(coords)
        hits = list({z["name"] for z, poly in zone_shapes if line.intersects(poly)})
        cj = await evaluate_carjacking_exposure(coords)
        dur_min = feat["properties"]["summary"]["duration"] / 60.0

        hard_penalty = 10000.0 * len(hits)
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
        waypoints = extract_corridor_waypoints(baseline_coords, best["feature"], max_waypoints=4)

    if base_eval["hits"] and len(best["hits"]) < len(base_eval["hits"]):
        avoided_list = [h for h in base_eval["hits"] if h not in best["hits"]]
        preview = ", ".join(avoided_list[:3]) + ("..." if len(avoided_list) > 3 else "")
        reason = f"⚠️ Desvio aplicado (evitou: {preview}) | +{round(best['duration_min'] - base_eval['duration_min'], 1)} min"
    elif best["index"] != 0:
        risk_diff = round(base_eval["carjacking"]["carjacking_score"] - best["carjacking"]["carjacking_score"], 1)
        time_diff = round(best["duration_min"] - base_eval["duration_min"], 1)
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
