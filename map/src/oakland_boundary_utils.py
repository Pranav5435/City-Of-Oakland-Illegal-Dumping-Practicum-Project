import json
from functools import lru_cache
from pathlib import Path

from shapely import covers, points
from shapely.geometry import Point, Polygon, box, shape
from shapely.ops import unary_union


SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent if SCRIPT_DIR.name == "src" else SCRIPT_DIR
BOUNDARY_FILE = BASE_DIR / "data" / "oakland_city_boundary.geojson"
# Deterministic lat/lon cutouts for non-Oakland bands that leak through the raw
# city polygon in the generated map. These are applied as simple rectangles so
# the rule stays stable and reproducible instead of hand-removing individual dots.
BOUNDARY_EXCLUSION_BOXES = (
    (37.7250, 37.8050, -122.1600, -122.1140),  # far-east lobe
    (37.7962, 37.7978, -122.3055, -122.3028),  # bridge outlier
    (37.7984, 37.7996, -122.2878, -122.2858),  # port outlier
    (37.8290, 37.8780, -122.2920, -122.2000),  # Berkeley / Orinda ridge
    (37.8798, 37.8838, -122.2485, -122.2285),  # north Berkeley row
    (37.8290, 37.8520, -122.2000, -122.1680),  # northeast Berkeley-side continuation
)
BOUNDARY_INCLUSION_POLYGONS = (
    (
        (-122.2700, 37.7910),
        (-122.2680, 37.7880),
        (-122.2620, 37.7760),
        (-122.2460, 37.7760),
        (-122.2460, 37.7910),
    ),
)


@lru_cache(maxsize=1)
def load_raw_oakland_boundary():
    payload = json.loads(BOUNDARY_FILE.read_text(encoding="utf-8"))
    features = payload.get("features") or []
    if not features:
        raise ValueError(f"Boundary file has no features: {BOUNDARY_FILE}")
    geometry = features[0].get("geometry")
    if not geometry:
        raise ValueError(f"Boundary file has no geometry: {BOUNDARY_FILE}")
    return shape(geometry)


@lru_cache(maxsize=1)
def load_boundary_exclusions():
    if not BOUNDARY_EXCLUSION_BOXES:
        return None
    return unary_union(
        [box(min_lon, min_lat, max_lon, max_lat) for min_lat, max_lat, min_lon, max_lon in BOUNDARY_EXCLUSION_BOXES]
    )


@lru_cache(maxsize=1)
def load_boundary_inclusions():
    if not BOUNDARY_INCLUSION_POLYGONS:
        return None
    return unary_union([Polygon(coords) for coords in BOUNDARY_INCLUSION_POLYGONS])


@lru_cache(maxsize=1)
def load_oakland_boundary():
    geometry = load_raw_oakland_boundary()
    inclusions = load_boundary_inclusions()
    if inclusions is not None:
        geometry = geometry.union(inclusions)
    exclusions = load_boundary_exclusions()
    if exclusions is None:
        return geometry
    return geometry.difference(exclusions)


@lru_cache(maxsize=1)
def get_oakland_bounds():
    min_lon, min_lat, max_lon, max_lat = load_oakland_boundary().bounds
    return min_lat, max_lat, min_lon, max_lon


def get_point_bounds(df, lat_col="latitude", lon_col="longitude", padding=0.003):
    if df.empty:
        return get_oakland_bounds()

    min_lat = float(df[lat_col].min()) - padding
    max_lat = float(df[lat_col].max()) + padding
    min_lon = float(df[lon_col].min()) - padding
    max_lon = float(df[lon_col].max()) + padding
    return min_lat, max_lat, min_lon, max_lon


def filter_to_oakland_boundary(df, lat_col="latitude", lon_col="longitude"):
    if df.empty:
        return df

    min_lat, max_lat, min_lon, max_lon = get_oakland_bounds()
    filtered = df[
        df[lat_col].between(min_lat, max_lat)
        & df[lon_col].between(min_lon, max_lon)
    ].copy()
    if filtered.empty:
        return filtered

    geometry = load_oakland_boundary()
    mask = covers(
        geometry,
        points(
            filtered[lon_col].astype(float).to_numpy(copy=False),
            filtered[lat_col].astype(float).to_numpy(copy=False),
        ),
    )
    return filtered.loc[mask].copy()


def is_in_oakland_boundary(lat, lon):
    min_lat, max_lat, min_lon, max_lon = get_oakland_bounds()
    if not (min_lat <= lat <= max_lat and min_lon <= lon <= max_lon):
        return False
    return load_oakland_boundary().covers(Point(float(lon), float(lat)))
