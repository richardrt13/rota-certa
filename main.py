import asyncio
import json
import math
import os
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from typing import List, Tuple
from zoneinfo import ZoneInfo
import asyncpg
import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from shapely.geometry import LineString, Point, shape
from shapely.ops import unary_union

load_dotenv()

DATABASE_URL = os.environ["DATABASE_URL"]
ORS_API_KEY = os.environ["ORS_API_KEY"]
ADMIN_PIN = os.environ.get("ADMIN_PIN", "")
ORS_DIRECTIONS_URL = "https://api.openrouteservice.org/v2/directions/driving-car/geojson"
OVERPASS_URL = "https://overpass-api.de/api/interpreter"
RIO_TZ = ZoneInfo("America/Sao_Paulo")

OVERPASS_QUERY = """
[out:json][timeout:30];
(
  nwr["railway"="station"](-23.082,-43.796,-22.746,-43.099);
  nwr["amenity"="police"](-23.082,-43.796,-22.746,-43.099);
  nwr["amenity"="hospital"](-23.082,-43.796,-22.746,-43.099);
);
out center;
"""

db_pool: asyncpg.Pool | None = None


async def ensure_and_seed_safe_pois():
    """Cria a tabela safe_pois se não existir e popula via OpenStreetMap caso esteja vazia."""
    if not db_pool:
        return
    try:
        async with db_pool.acquire() as conn:
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS safe_pois (
                    id SERIAL PRIMARY KEY,
                    external_id VARCHAR(100) UNIQUE,
                    name VARCHAR(150) NOT NULL,
                    category VARCHAR(50) NOT NULL,
                    geom GEOMETRY(Point, 4326) NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_safe_pois_geom ON safe_pois USING GIST (geom);
                """
            )
            count = await conn.fetchval("SELECT COUNT(*) FROM safe_pois;")
            if count and count > 0:
                return

        async with httpx.AsyncClient(timeout=45.0) as client:
            resp = await client.post(OVERPASS_URL, data={"data": OVERPASS_QUERY})
            resp.raise_for_status()
            elements = resp.json().get("elements", [])

        async with db_pool.acquire() as conn:
            for el in elements:
                tags = el.get("tags", {})
                lat = el.get("lat") or (el.get("center") or {}).get("lat")
                lng = el.get("lon") or (el.get("center") or {}).get("lon")
                if lat is None or lng is None:
                    continue

                if tags.get("railway") == "station":
                    category, default_name = "station", "Estação Metrô/Trem"
                elif tags.get("amenity") == "police":
                    category, default_name = "police", "Unidade Policial"
                elif tags.get("amenity") == "hospital":
                    category, default_name = "hospital", "Hospital"
                else:
                    continue

                name = (tags.get("name") or default_name)[:145]
                ext_id = f"osm_{el['type']}_{el['id']}"
                await conn.execute(
                    """
                    INSERT INTO safe_pois (external_id, name, category, geom)
                    VALUES ($1, $2, $3, ST_SetSRID(ST_MakePoint($4, $5), 4326))
                    ON CONFLICT (external_id) DO NOTHING;
                    """,
                    ext_id,
                    name,
                    category,
                    float(lng),
                    float(lat),
                )
    except Exception as exc:
        print(f"Aviso ao sincronizar safe_pois: {exc}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global db_pool
    db_pool = await asyncpg.create_pool(DATABASE_URL, min_size=1, max_size=5)
    asyncio.create_task(ensure_and_seed_safe_pois())
    yield
    await db_pool.close()


app = FastAPI(title="Rota Certa RJ - V2.2", lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(status_code=500, content={"detail": f"Erro interno: {str(exc)}"})


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
        return 1.35
    if 18 <= hour < 21 or 5 <= hour < 7:
        return 1.10
    return 0.85


def haversine_km(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    r = 6371.0
    lat1, lon1 = math.radians(p1[0]), math.radians(p1[1])
    lat2, lon2 = math.radians(p2[0]), math.radians(p2[1])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


async def discover_dynamic_safe_hubs(
    origin: Tuple[float, float], destination: Tuple[float, float]
) -> List[Tuple[str, Tuple[float, float]]]:
    direct_km = haversine_km(origin, destination)
    if direct_km < 6.0:
        return []

    try:
        async with db_pool.acquire() as conn:
            rows = await conn.fetch(
                """
                WITH endpoints AS (
                    SELECT
                        ST_Transform(ST_SetSRID(ST_MakePoint($1, $2), 4326), 31983) AS p1,
                        ST_Transform(ST_SetSRID(ST_MakePoint($3, $4), 4326), 31983) AS p2
                ),
                vectors AS (
                    SELECT
                        ST_X(p2) - ST_X(p1) AS dx,
                        ST_Y(p2) - ST_Y(p1) AS dy,
                        (ST_X(p1) + ST_X(p2)) / 2.0 AS mx,
                        (ST_Y(p1) + ST_Y(p2)) / 2.0 AS my
                    FROM endpoints
                ),
                lateral_targets AS (
                    SELECT 'Flanco A' AS side,
                           ST_Transform(ST_SetSRID(ST_MakePoint(mx - 0.30 * dy, my + 0.30 * dx), 31983), 4326) AS target_geom
                    FROM vectors
                    UNION ALL
                    SELECT 'Flanco B' AS side,
                           ST_Transform(ST_SetSRID(ST_MakePoint(mx + 0.30 * dy, my - 0.30 * dx), 31983), 4326) AS target_geom
                    FROM vectors
                )
                SELECT DISTINCT ON (lt.side)
                    sp.name AS hub_name,
                    ST_Y(sp.geom) AS lat,
                    ST_X(sp.geom) AS lng,
                    pr.risk_index
                FROM lateral_targets lt
                JOIN safe_pois sp
                  ON ST_DWithin(sp.geom::geography, lt.target_geom::geography, 4500)
                JOIN precinct_risk pr
                  ON ST_Intersects(sp.geom, pr.geom)
                WHERE pr.risk_index < 55.0
                  AND NOT EXISTS (
                      SELECT 1 FROM avoid_zones az
                      WHERE (az.expires_at IS NULL OR az.expires_at > NOW())
                        AND ST_DWithin(sp.geom::geography, az.geom::geography, 250)
                  )
                ORDER BY lt.side, pr.risk_index ASC, ST_Distance(sp.geom::geography, lt.target_geom::geography) ASC;
                """,
                origin[1], origin[0], destination[1], destination[0],
            )
    except Exception:
        return []

    hubs: List[Tuple[str, Tuple[float, float]]] = []
    seen_names = set()
    for r in rows:
        hub_pt = (float(r["lat"]), float(r["lng"]))
        via_km = haversine_km(origin, hub_pt) + haversine_km(hub_pt, destination)
        if via_km / max(direct_km, 0.1) <= 1.75 and r["hub_name"] not in seen_names:
            seen_names.add(r["hub_name"])
            hubs.append((r["hub_name"], hub_pt))

    return hubs


async def fetch_zones_near_corridors(candidate_lines: List[LineString]) -> List[dict]:
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


def analyze_road_hierarchy_and_speed(feature: dict) -> dict:
    coords = feature["geometry"]["coordinates"]
    props = feature.get("properties", {})
    waytypes = props.get("extras", {}).get("waytype", {}).get("values", [])

    arterial_deg = 0.0
    local_deg = 0.0
    total_deg = 0.0

    if waytypes and len(coords) >= 2:
        for start_i, end_i, code in waytypes:
            length = LineString(coords[start_i : end_i + 1]).length
            total_deg += length
            if code == 1:
                arterial_deg += length
            elif code >= 3:
                local_deg += length

    # Calcula proporção de trechos lentos (< 22 km/h) direto dos steps nativos do ORS
    slow_dist = 0.0
    total_step_dist = 0.0
    for seg in props.get("segments", []):
        for step in seg.get("steps", []):
            d_m = float(step.get("distance", 0.0))
            t_s = float(step.get("duration", 0.0))
            if d_m > 0 and t_s > 0:
                total_step_dist += d_m
                speed_kmh = (d_m / 1000.0) / (t_s / 3600.0)
                if speed_kmh < 22.0:
                    slow_dist += d_m

    return {
        "arterial_share": (arterial_deg / total_deg) if total_deg > 0 else 0.5,
        "local_share": (local_deg / total_deg) if total_deg > 0 else 0.1,
        "slow_share": (slow_dist / total_step_dist) if total_step_dist > 0 else 0.1,
    }


async def evaluate_route_safety(feature: dict) -> dict:
    route_coords = feature["geometry"]["coordinates"]
    total_km = max(feature["properties"]["summary"]["distance"] / 1000.0, 0.2)
    wkt_line = "LINESTRING(" + ", ".join(f"{lng} {lat}" for lng, lat in route_coords) + ")"
    time_mult = get_carjacking_time_multiplier()

    async with db_pool.acquire() as conn:
        cisp_rows = await conn.fetch(
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

        border_rows = await conn.fetch(
            """
            SELECT
                name,
                ST_Length(
                    ST_Intersection(
                        ST_GeomFromText($1, 4326)::geography,
                        ST_Buffer(geom::geography, 180)
                    )
                ) AS border_meters
            FROM avoid_zones
            WHERE (expires_at IS NULL OR expires_at > NOW())
              AND ST_DWithin(ST_GeomFromText($1, 4326)::geography, geom::geography, 180);
            """,
            wkt_line,
        )

        try:
            poi_rows = await conn.fetch(
                """
                SELECT category, COUNT(*) AS qty
                FROM safe_pois
                WHERE ST_DWithin(geom::geography, ST_GeomFromText($1, 4326)::geography, 250)
                GROUP BY category;
                """,
                wkt_line,
            )
        except Exception:
            poi_rows = []

    weighted_sum = 0.0
    covered_km = 0.0
    peak_cisp_risk = 0.0
    warnings = []

    for r in cisp_rows:
        km_exposed = r["meters_in_cisp"] / 1000.0
        risk_idx = float(r["risk_index"])
        weighted_sum += km_exposed * risk_idx
        covered_km += km_exposed
        if km_exposed >= 0.4 and risk_idx > peak_cisp_risk:
            peak_cisp_risk = risk_idx
        if risk_idx >= 55.0 and km_exposed >= 0.5:
            warnings.append(
                f"{r['precinct_name']} ({r['roubo_veiculo_6m']} roubos/6m, {km_exposed:.1f} km)"
            )

    avg_cisp_risk = (weighted_sum / max(covered_km, 0.1)) if covered_km > 0 else 20.0
    base_precinct_score = (0.65 * avg_cisp_risk) + (0.35 * peak_cisp_risk)

    total_border_km = sum(b["border_meters"] / 1000.0 for b in border_rows)
    border_ratio = min(total_border_km / total_km, 1.0)
    border_penalty = min(35.0, border_ratio * 150.0)

    flow = analyze_road_hierarchy_and_speed(feature)
    road_adjustment = (
        (flow["local_share"] * 25.0)
        + (flow["slow_share"] * 14.0)
        - (flow["arterial_share"] * 12.0)
    )

    poi_weights = {"police": 3.0, "station": 2.0, "hospital": 1.5}
    raw_poi_bonus = sum(poi_weights.get(p["category"], 1.0) * p["qty"] for p in poi_rows)
    eyes_on_street_bonus = min(15.0, raw_poi_bonus)
    total_safe_pois = sum(p["qty"] for p in poi_rows)

    calibrated_score = (
        base_precinct_score + border_penalty + road_adjustment - eyes_on_street_bonus
    ) * time_mult
    final_score = round(max(0.0, min(calibrated_score, 100.0)), 1)

    return {
        "carjacking_score": final_score,
        "arterial_pct": int(flow["arterial_share"] * 100),
        "safe_pois_count": total_safe_pois,
        "border_km": round(total_border_km, 1),
        "time_multiplier": time_mult,
        "high_risk_precincts": warnings,
    }


async def call_ors(
    client: httpx.AsyncClient,
    coords_lonlat: List[List[float]],
    avoid_multipolygon: dict | None = None,
    request_alternatives: bool = False,
) -> dict:
    headers = {"Authorization": ORS_API_KEY, "Content-Type": "application/json"}
    body: dict = {
        "coordinates": coords_lonlat,
        "radiuses": [1500] * len(coords_lonlat),
        "instructions": True,
        "extra_info": ["waytype"],
    }
    if avoid_multipolygon:
        body["options"] = {"avoid_polygons": avoid_multipolygon}
    elif request_alternatives and len(coords_lonlat) == 2:
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
    direct_coords = [[req.origin[1], req.origin[0]], [req.destination[1], req.destination[0]]]
    dynamic_hubs = await discover_dynamic_safe_hubs(req.origin, req.destination)

    async with httpx.AsyncClient(timeout=20.0) as client:
        initial_tasks = [call_ors(client, direct_coords, None, request_alternatives=True)]
        for _, hub in dynamic_hubs:
            via_coords = [
                [req.origin[1], req.origin[0]],
                [hub[1], hub[0]],
                [req.destination[1], req.destination[0]],
            ]
            initial_tasks.append(call_ors(client, via_coords, None, request_alternatives=False))

        initial_results = await asyncio.gather(*initial_tasks, return_exceptions=True)

        alt_res = initial_results[0]
        if isinstance(alt_res, Exception):
            raise HTTPException(status_code=502, detail=str(alt_res))
        if not alt_res.get("features"):
            raise HTTPException(status_code=502, detail="Erro ao obter rotas iniciais.")

        alt_features = alt_res["features"]
        baseline_feat = alt_features[0]
        baseline_coords = baseline_feat["geometry"]["coordinates"]

        tagged_candidates: List[Tuple[str, dict, Tuple[float, float] | None]] = [
            ("Direta", f, None) for f in alt_features
        ]
        for idx, res in enumerate(initial_results[1:]):
            if not isinstance(res, Exception) and res.get("features"):
                hub_name, hub_pt = dynamic_hubs[idx]
                tagged_candidates.append((f"Via {hub_name}", res["features"][0], hub_pt))

        candidate_lines = [LineString(f["geometry"]["coordinates"]) for _, f, _ in tagged_candidates]
        zones = await fetch_zones_near_corridors(candidate_lines)
        zone_shapes = [(z, shape(z["geometry"])) for z in zones]

        if zones:
            avoid_mp = {
                "type": "MultiPolygon",
                "coordinates": [z["geometry"]["coordinates"] for z in zones],
            }
            avoid_tasks = []
            avoid_meta = []

            if any(candidate_lines[0].intersects(poly) for _, poly in zone_shapes):
                avoid_tasks.append(call_ors(client, direct_coords, avoid_mp))
                avoid_meta.append(("Direta sem comunidades", None))

            for label, feat, hub_pt in tagged_candidates:
                if hub_pt is not None:
                    line = LineString(feat["geometry"]["coordinates"])
                    if any(line.intersects(poly) for _, poly in zone_shapes):
                        via_coords = [
                            [req.origin[1], req.origin[0]],
                            [hub_pt[1], hub_pt[0]],
                            [req.destination[1], req.destination[0]],
                        ]
                        avoid_tasks.append(call_ors(client, via_coords, avoid_mp))
                        avoid_meta.append((f"{label} (sem comunidades)", hub_pt))

            if avoid_tasks:
                avoid_results = await asyncio.gather(*avoid_tasks, return_exceptions=True)
                for (lbl, hub_pt), a_res in zip(avoid_meta, avoid_results):
                    if not isinstance(a_res, Exception) and a_res.get("features"):
                        tagged_candidates.append((lbl, a_res["features"][0], hub_pt))

    evaluated = []
    for idx, (label, feat, _) in enumerate(tagged_candidates):
        coords = feat["geometry"]["coordinates"]
        line = LineString(coords)
        hits = list({z["name"] for z, poly in zone_shapes if line.intersects(poly)})
        safety = await evaluate_route_safety(feat)
        dur_min = feat["properties"]["summary"]["duration"] / 60.0
        dist_km = feat["properties"]["summary"]["distance"] / 1000.0

        hard_penalty = 10000.0 * len(hits)
        exposure_factor = max(1.0, dist_km / 12.0)
        combined_cost = hard_penalty + dur_min + (safety["carjacking_score"] * exposure_factor * 0.45)

        evaluated.append({
            "index": idx,
            "label": label,
            "feature": feat,
            "coords": coords,
            "hits": hits,
            "safety": safety,
            "duration_min": dur_min,
            "combined_cost": combined_cost,
        })

    best = min(evaluated, key=lambda x: x["combined_cost"])
    base_eval = evaluated[0]

    waypoints = []
    if best["index"] != 0:
        waypoints = extract_corridor_waypoints(baseline_coords, best["feature"], max_waypoints=4)

    risk_diff = round(base_eval["safety"]["carjacking_score"] - best["safety"]["carjacking_score"], 1)
    time_diff = round(best["duration_min"] - base_eval["duration_min"], 1)
    context_info = f"{best['safety']['arterial_pct']}% arteriais • {best['safety']['safe_pois_count']} polos seguros"

    if best["index"] != 0:
        reason = (
            f"🛡️ Escolhida: {best['label']} ({context_info} • "
            f"-{max(0.0, risk_diff)} pts de risco, {time_diff:+.1f} min)"
        )
    else:
        reason = f"✅ Rota padrão já é a mais segura ({context_info})"

    summary = best["feature"]["properties"]["summary"]
    return {
        "detour_applied": best["index"] != 0,
        "selection_reason": reason,
        "avoided_zones": base_eval["hits"],
        "duration_minutes": round(summary["duration"] / 60, 1),
        "distance_km": round(summary["distance"] / 1000, 2),
        "carjacking": best["safety"],
        "injected_waypoints": waypoints,
        "baseline_geojson": baseline_feat,
        "safe_geojson": best["feature"],
        "google_maps_url": build_google_maps_url(req.origin, req.destination, waypoints),
    }
