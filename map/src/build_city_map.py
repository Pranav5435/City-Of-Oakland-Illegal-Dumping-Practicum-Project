import os
import json
from pathlib import Path

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from oakland_boundary_utils import filter_to_oakland_boundary, get_point_bounds

SCRIPT_DIR      = Path(__file__).resolve().parent
BASE_DIR        = SCRIPT_DIR.parent if SCRIPT_DIR.name == "src" else SCRIPT_DIR
PUBLIC_DIR      = BASE_DIR / "public"
OUTPUT_FILE     = PUBLIC_DIR / "oakland_city_map.html"

URL             = "https://data.oaklandca.gov/resource/dmqt-4g4v.json"
BATCH_SIZE      = 2000
MAX_RECORDS     = int(os.getenv("MAX_RECORDS",        "0"))
MAX_RENDER      = int(os.getenv("MAX_RENDER_POINTS",  "20000"))
TILE_STYLE      = os.getenv("TILE_STYLE",             "dark")   # dark | light | osm
GRID_SIZE       = float(os.getenv("GRID_SIZE_DEG",    "0.0025"))
RENDER_GRID     = float(os.getenv("RENDER_GRID_SIZE_DEG", "0.005"))
SAMPLE_SEED     = int(os.getenv("RENDER_SAMPLE_SEED", "42"))

TILE_URLS = {
    "dark":  "https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png",
    "light": "https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png",
    "osm":   "https://tile.openstreetmap.org/{z}/{x}/{y}.png",
}
TILE_URL = TILE_URLS.get(TILE_STYLE, TILE_URLS["dark"])

# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------

session = requests.Session()
session.mount("https://", HTTPAdapter(max_retries=Retry(
    total=4, connect=4, read=4, backoff_factor=0.6,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["GET"],
)))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def spread_sample(df, max_points, lat_col, lon_col, grid_size, seed, weight_col=None):
    if len(df) <= max_points:
        return df.copy()
    s = df.copy()
    s["_rlat"] = (s[lat_col] / grid_size).round().astype(int)
    s["_rlon"] = (s[lon_col] / grid_size).round().astype(int)
    one = s.groupby(["_rlat", "_rlon"], sort=False, group_keys=False).sample(n=1, random_state=seed)
    if len(one) >= max_points:
        return one.sample(n=max_points, random_state=seed).drop(columns=["_rlat", "_rlon"])
    remaining = s.drop(index=one.index)
    slots = max_points - len(one)
    if slots > 0 and not remaining.empty:
        weights = remaining[weight_col] if weight_col and weight_col in remaining.columns else None
        extra = remaining.sample(n=min(slots, len(remaining)), random_state=seed, weights=weights)
        s = pd.concat([one, extra])
    else:
        s = one
    return s.drop(columns=["_rlat", "_rlon"])


def marker_color(pct_rank: float) -> str:
    if pct_rank <= 0.33: return "#39FF14"   # low  — neon green
    if pct_rank <= 0.66: return "#FFFF00"   # med  — yellow
    return "#FF3131"                         # high — red


def marker_label(pct_rank: float) -> str:
    if pct_rank <= 0.33: return "LOW"
    if pct_rank <= 0.66: return "MEDIUM"
    return "HIGH"


def build_html(render_df: pd.DataFrame, bounds: tuple) -> str:
    min_lat, max_lat, min_lon, max_lon = bounds

    # One compact JSON array — the only per-point data the HTML needs
    markers = [
        {
            "lat": round(float(r.lat), 6),
            "lon": round(float(r.lon), 6),
            "c":   marker_color(r.density_rank),
            "l":   marker_label(r.density_rank),
        }
        for r in render_df.itertuples(index=False)
    ]
    markers_json = json.dumps(markers, separators=(",", ":"))

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Oakland Illegal Dumping Map</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/leaflet@1.9.3/dist/leaflet.css"/>
<style>
html,body,#map{{height:100%;margin:0;padding:0;}}
.leaflet-popup-content-wrapper{{background:#000!important;color:#fff!important;border:1px solid #444;border-radius:0!important;}}
.popup-card{{font-family:monospace;font-size:12px;}}
.popup-row{{margin-bottom:4px;}}
</style>
</head>
<body>
<div id="map"></div>
<script src="https://cdn.jsdelivr.net/npm/leaflet@1.9.3/dist/leaflet.js"></script>
<script>
const map = L.map('map', {{preferCanvas: true}});
map.fitBounds([[{min_lat},{min_lon}],[{max_lat},{max_lon}]]);

L.tileLayer('{TILE_URL}', {{
  attribution: '&copy; OpenStreetMap contributors &copy; CARTO',
  subdomains: 'abcd',
  maxZoom: 20
}}).addTo(map);

// Reverse geocode cache
const cache = new Map();
async function reverseStreet(lat, lon) {{
  const key = lat.toFixed(6) + ',' + lon.toFixed(6);
  if (cache.has(key)) return cache.get(key);
  try {{
    const url = 'https://geocode.arcgis.com/arcgis/rest/services/World/GeocodeServer/reverseGeocode'
              + '?f=pjson&location=' + encodeURIComponent(lon + ',' + lat);
    const d = await fetch(url).then(r => r.json());
    const v = d?.address?.ShortLabel || d?.address?.Address || d?.address?.Match_addr
              || lat.toFixed(5) + ', ' + lon.toFixed(5);
    cache.set(key, v);
    return v;
  }} catch {{ return lat.toFixed(5) + ', ' + lon.toFixed(5); }}
}}

// Render all markers from a single JSON array
const MARKERS = {markers_json};
MARKERS.forEach(({{"lat":lat,"lon":lon,"c":c,"l":l}}) => {{
  L.circleMarker([lat, lon], {{
    radius: 7, fillColor: c, fillOpacity: 1, stroke: false
  }}).addTo(map).bindPopup(
    '<div class="popup-card">'
    + '<div class="popup-row"><b>Street:</b> '
    + '<span class="lazy-addr" data-lat="' + lat + '" data-lon="' + lon + '">Loading...</span>'
    + '</div>'
    + '<div><b>Likelihood:</b> ' + l + '</div>'
    + '</div>',
    {{maxWidth: 280}}
  );
}});

// Lazy-load street address on popup open
map.on('popupopen', async e => {{
  const el = e.popup.getElement()?.querySelector('.lazy-addr');
  if (!el || el.dataset.loaded) return;
  el.dataset.loaded = '1';
  el.textContent = await reverseStreet(parseFloat(el.dataset.lat), parseFloat(el.dataset.lon));
}});
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

try:
    PUBLIC_DIR.mkdir(parents=True, exist_ok=True)

    print("1. Pulling Oakland data...")
    all_rows, offset = [], 0
    while True:
        if MAX_RECORDS > 0 and offset >= MAX_RECORDS:
            break
        limit = BATCH_SIZE if MAX_RECORDS <= 0 else min(BATCH_SIZE, MAX_RECORDS - offset)
        print(f"   Fetching rows {offset:,} – {offset + limit - 1:,}...")
        r = session.get(URL, params={"$limit": limit, "$offset": offset, "$select": "srx,sry"}, timeout=(5, 25))
        r.raise_for_status()
        batch = r.json()
        if not batch:
            break
        all_rows.extend(batch)
        offset += len(batch)
        if len(batch) < limit:
            break

    df = pd.DataFrame(all_rows).rename(columns={"srx": "lon", "sry": "lat"})
    df[["lat", "lon"]] = df[["lat", "lon"]].apply(pd.to_numeric, errors="coerce")
    df = df.dropna(subset=["lat", "lon"])

    before = len(df)
    df = filter_to_oakland_boundary(df, lat_col="lat", lon_col="lon")
    dropped = before - len(df)
    if dropped:
        print(f"   Dropped {dropped:,} out-of-bound points.")
    if df.empty:
        raise ValueError("No valid location data returned from API.")

    # Density scoring
    df["lat_bin"] = (df["lat"] / GRID_SIZE).round().astype(int)
    df["lon_bin"] = (df["lon"] / GRID_SIZE).round().astype(int)
    cell_counts   = df.groupby(["lat_bin", "lon_bin"]).size().rename("cell_count")
    df            = df.join(cell_counts, on=["lat_bin", "lon_bin"])
    lookup        = cell_counts.to_dict()

    def neighborhood_density(lb, ob):
        return sum(lookup.get((lb + dlat, ob + dlon), 0) for dlat in (-1,0,1) for dlon in (-1,0,1))

    df["local_density"] = [neighborhood_density(lb, ob) for lb, ob in zip(df["lat_bin"], df["lon_bin"])]
    df["density_rank"]  = df["local_density"].rank(pct=True, method="average")

    # Sample down for rendering
    render_df = spread_sample(df, MAX_RENDER, "lat", "lon", RENDER_GRID, SAMPLE_SEED, "local_density")
    render_df["lat"] = render_df["lat"].round(6)
    render_df["lon"] = render_df["lon"].round(6)
    render_df = filter_to_oakland_boundary(render_df, lat_col="lat", lon_col="lon")

    print(f"2. Building map ({len(render_df):,} rendered points from {len(df):,} fetched)...")
    bounds   = get_point_bounds(render_df, lat_col="lat", lon_col="lon")
    html_str = build_html(render_df, bounds)
    OUTPUT_FILE.write_text(html_str, encoding="utf-8")
    print(f"   Wrote {OUTPUT_FILE}  ({len(html_str):,} chars)")
    print("SUCCESS: Open 'public/oakland_city_map.html' in your browser.")

except KeyboardInterrupt:
    print("\nStopped by user.")
except Exception as e:
    print(f"Error: {e}")
finally:
    session.close()