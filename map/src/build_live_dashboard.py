import argparse
import json
import os
import subprocess
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from oakland_boundary_utils import filter_to_oakland_boundary, get_point_bounds, is_in_oakland_boundary

# --- CONFIG ---
API_KEY           = "c25d0e24a0764c8ca0f9240667be3420"
OAKLAND_DATA_ID   = "dmqt-4g4v"
SCRIPT_DIR        = Path(__file__).resolve().parent
BASE_DIR          = SCRIPT_DIR.parent if SCRIPT_DIR.name == "src" else SCRIPT_DIR
DATA_DIR          = BASE_DIR / "data"
PUBLIC_DIR        = BASE_DIR / "public"
LOCAL_DB          = DATA_DIR / "user_reports.json"
CITY_CACHE        = DATA_DIR / "city_cache.json"
PORT              = 8080
CACHE_MAX_AGE_HRS = float(os.getenv("CACHE_MAX_AGE_HOURS", "24"))
CACHE_REFRESH_INTERVAL_SECONDS = max(60, int(float(os.getenv("CACHE_REFRESH_INTERVAL_SECONDS", "300"))))
DASHBOARD_FILE_NAME = "oakland_dashboard.html"
DASHBOARD_FILE    = PUBLIC_DIR / DASHBOARD_FILE_NAME

API_CONNECT_TIMEOUT= float(os.getenv("API_CONNECT_TIMEOUT","4"))
API_READ_TIMEOUT   = float(os.getenv("API_READ_TIMEOUT",   "10"))
FETCH_BATCH_SIZE   = int(os.getenv("FETCH_BATCH_SIZE",     "50000"))
MAX_RENDER_POINTS  = int(os.getenv("MAX_RENDER_POINTS",    "20000"))
RISK_GRID_SIZE     = float(os.getenv("RISK_GRID_SIZE_DEG", "0.0025"))
RENDER_GRID_SIZE   = float(os.getenv("RENDER_GRID_SIZE_DEG","0.005"))
RENDER_SAMPLE_SEED = int(os.getenv("RENDER_SAMPLE_SEED",   "42"))

address_cache        = {}
rebuild_lock         = threading.Lock()
report_count_lock    = threading.Lock()
latest_total_reports = 0

if not os.path.exists(LOCAL_DB):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOCAL_DB, 'w') as f:
        json.dump([], f)


# ---------------------------------------------------------------------------
# City data cache
# ---------------------------------------------------------------------------

def get_city_cache_age_hours() -> float | None:
    if not CITY_CACHE.exists():
        return None
    return (time.time() - CITY_CACHE.stat().st_mtime) / 3600


def is_city_cache_stale() -> bool:
    age_hrs = get_city_cache_age_hours()
    if age_hrs is None:
        return True
    return age_hrs > CACHE_MAX_AGE_HRS

def is_rebuild_needed() -> bool:
    markers_path = PUBLIC_DIR / "data" / "markers.json"
    if not markers_path.exists():
        return True
    if not DASHBOARD_FILE.exists():
        return True
    script_mtime = Path(__file__).stat().st_mtime
    if script_mtime > markers_path.stat().st_mtime:
        return True
    if script_mtime > DASHBOARD_FILE.stat().st_mtime:
        return True
    if is_city_cache_stale():
        return True
    m = markers_path.stat().st_mtime
    if CITY_CACHE.exists() and CITY_CACHE.stat().st_mtime > m:
        return True
    if LOCAL_DB.exists() and LOCAL_DB.stat().st_mtime > m:
        return True
    return False


def load_city_cache() -> list | None:
  """Return cached rows if they exist and are fresh, otherwise None."""
  age_hrs = get_city_cache_age_hours()
  if age_hrs is None:
    return None
  if age_hrs > CACHE_MAX_AGE_HRS:
    print(f"   Cache is {age_hrs:.1f}h old (limit {CACHE_MAX_AGE_HRS}h) - refreshing from API...")
    return None
  print(f"   Using cached city data ({age_hrs:.1f}h old, limit {CACHE_MAX_AGE_HRS}h).")
  with open(CITY_CACHE) as f:
    return json.load(f)


def save_city_cache(rows: list):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(CITY_CACHE, 'w') as f:
        json.dump(rows, f)


# ---------------------------------------------------------------------------
# Network helpers
# ---------------------------------------------------------------------------

def create_retry_session():
    session = requests.Session()
    retry = Retry(
        total=3, connect=3, read=3, backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=8, pool_maxsize=8)
    session.mount("https://", adapter)
    return session


def get_address_from_coords(lat, lon):
    key = (round(lat, 5), round(lon, 5))
    if key in address_cache:
        return address_cache[key]
    try:
        url = (f"https://api.geoapify.com/v1/geocode/reverse"
               f"?lat={lat}&lon={lon}&apiKey={API_KEY}")
        r = requests.get(url, timeout=2).json()
        props = r['features'][0]['properties']
        value = props.get('street', props.get('name', f"Sector {lat:.3f}"))
    except Exception:
        value = f"Sector {lat:.3f}"
    address_cache[key] = value
    return value


# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------

def fetch_all_city_rows(session, base_url):
    rows, offset = [], 0
    while True:
        resp = session.get(
            base_url,
            params={"$select": "srx,sry", "$limit": FETCH_BATCH_SIZE, "$offset": offset},
            timeout=(API_CONNECT_TIMEOUT, API_READ_TIMEOUT),
        )
        resp.raise_for_status()
        batch = resp.json()
        if not batch:
            break
        rows.extend(batch)
        offset += len(batch)
        if len(batch) < FETCH_BATCH_SIZE:
            break
    return rows


# ---------------------------------------------------------------------------
# Sampling / risk scoring
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


def assign_density_risk_scores(df, lat_col="latitude", lon_col="longitude"):
    scored = df.copy()
    if scored.empty:
        for col in ("local_density", "density_rank", "risk_pct"):
            scored[col] = pd.Series(dtype="float64")
        return scored
    scored["_rlat"] = (scored[lat_col] / RISK_GRID_SIZE).round().astype(int)
    scored["_rlon"] = (scored[lon_col] / RISK_GRID_SIZE).round().astype(int)
    cell_counts = scored.groupby(["_rlat", "_rlon"]).size().rename("cell_count")
    scored = scored.join(cell_counts, on=["_rlat", "_rlon"])
    lookup = cell_counts.to_dict()

    def neighborhood_density(lb, lnb):
        return sum(lookup.get((lb + dlat, lnb + dlon), 0) for dlat in (-1, 0, 1) for dlon in (-1, 0, 1))

    scored["local_density"] = [neighborhood_density(lb, lnb) for lb, lnb in zip(scored["_rlat"], scored["_rlon"])]
    scored["density_rank"] = 0.5 if (len(scored) == 1 or scored["local_density"].nunique() == 1) \
                             else scored["local_density"].rank(pct=True, method="average")
    scored["risk_pct"] = (10 + (scored["density_rank"] * 89)).round().clip(10, 99).astype(int)
    return scored


# ---------------------------------------------------------------------------
# HTML template — pure Leaflet, no Folium
# ---------------------------------------------------------------------------

def _point_color(risk_pct: int, is_user: bool) -> str:
    if is_user:        return "#00D1FF"
    if risk_pct > 80:  return "#FF3131"
    if risk_pct >= 40: return "#FFFF00"
    return "#39FF14"


def render_map_html(
    top_high_rows: pd.DataFrame,
    top_low_rows: pd.DataFrame,
    total_reports: int,
    map_bounds: tuple,
    markers_json_file: str,
) -> str:
    min_lat, max_lat, min_lon, max_lon = map_bounds

    def build_mode_data(rows, score_color):
        leaderboard = ""
        popup_data  = []
        for r in rows.itertuples(index=False):
            addr = str(r.address).strip() or get_address_from_coords(r.latitude, r.longitude)
            safe = addr.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            leaderboard += (
                f'<div style="font-size:10px;margin-bottom:8px;border-bottom:1px solid #222;padding-bottom:4px;">'
                f'<b style="color:{score_color};">{r.risk_pct}%</b> {safe}</div>'
            )
            popup_data.append({
                "lat":     round(float(r.latitude),  6),
                "lon":     round(float(r.longitude), 6),
                "risk":    int(r.risk_pct),
                "address": safe,
                "color":   _point_color(int(r.risk_pct), r.source == "user"),
            })
        return leaderboard, popup_data

    high_leaderboard_html, high_popup_data = build_mode_data(top_high_rows, "#FF3131")
    low_leaderboard_html,  low_popup_data  = build_mode_data(top_low_rows,  "#39FF14")
    high_top_json       = json.dumps(high_popup_data).replace("</", "<\\/")
    low_top_json        = json.dumps(low_popup_data).replace("</", "<\\/")
    high_leaderboard_js = json.dumps(high_leaderboard_html)
    low_leaderboard_js  = json.dumps(low_leaderboard_html)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Oakland Illegal Dumping</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/leaflet@1.9.3/dist/leaflet.css"/>
<style>
html,body,#map{{height:100%;margin:0;padding:0;}}
.leaflet-popup-content-wrapper{{background:#000!important;color:#fff!important;border:1px solid #444;border-radius:0!important;}}
.panel{{position:fixed;background:rgba(0,0,0,.95);color:#fff;padding:20px;z-index:9999;font-family:monospace;border-radius:4px;border:1px solid #333;}}
.back-btn{{position:fixed;top:20px;left:20px;z-index:10000;padding:8px 12px;border:1px solid #00D1FF;border-radius:6px;background:rgba(0,0,0,.92);color:#00D1FF;font-family:monospace;font-size:11px;font-weight:bold;text-decoration:none;cursor:pointer;}}
.back-btn:hover{{border-color:#6fe6ff;color:#6fe6ff;}}
.top-ten{{top:72px;left:20px;width:280px;border-top:4px solid #FF3131;}}
.report-form{{bottom:20px;left:20px;width:320px;border-bottom:4px solid #00D1FF;}}
input{{background:#111;border:1px solid #333;color:#fff;width:100%;padding:12px;margin:8px 0;outline:none;box-sizing:border-box;}}
button{{width:100%;padding:12px;background:#00D1FF;color:#000;font-weight:bold;border:none;cursor:pointer;}}
#loading{{position:fixed;top:50%;left:50%;transform:translate(-50%,-50%);background:rgba(0,0,0,.85);
          color:#00D1FF;font-family:monospace;padding:20px 32px;border:1px solid #333;z-index:99999;}}
.legend-drawer{{position:fixed;right:20px;bottom:20px;z-index:9999;font-family:monospace;}}
.legend-toggle{{background:rgba(0,0,0,.92);color:#00D1FF;border:1px solid #00D1FF;padding:10px 12px;font-size:10px;cursor:pointer;font-weight:bold;}}
.legend-panel{{margin-top:8px;width:220px;background:rgba(0,0,0,.94);border:1px solid #333;padding:12px;max-height:0;overflow:hidden;opacity:0;transform:translateY(8px);transition:max-height .25s ease,opacity .2s ease,transform .2s ease;}}
.legend-drawer.open .legend-panel{{max-height:220px;opacity:1;transform:translateY(0);}}
.legend-title{{font-size:10px;color:#ccc;margin-bottom:8px;letter-spacing:.4px;}}
.legend-row{{display:flex;align-items:center;gap:8px;font-size:10px;color:#ddd;margin:5px 0;}}
.legend-dot{{width:10px;height:10px;border-radius:50%;border:1px solid rgba(255,255,255,.4);display:inline-block;}}
</style>
</head>
<body>
<div id="loading">Loading map data…</div>
<div id="map"></div>
<a href="#" class="back-btn" onclick="history.back(); return false;">← Back</a>

<div class="panel top-ten" id="mode_panel">
  <div id="mode_title" style="font-size:10px;color:#FF3131;font-weight:bold;margin-bottom:10px;">TOP 10 RISK LOCATIONS</div>
  <div id="mode_list">{high_leaderboard_html}</div>
  <div style="margin-top:10px;">
    <a href="#" id="mode_toggle" style="color:#00D1FF;font-size:10px;text-decoration:none;">OPEN LEAST-LIKELY VIEW</a>
  </div>
</div>

<div class="panel report-form">
  <div style="font-size:10px;color:#00D1FF;font-weight:bold;">Enter location of illegal dumping occurrence</div>
  <input id="place_in" placeholder="Street or Landmark..."/>
  <button id="gen_btn" onclick="submitPlace()">Enter</button>
  <div id="submit_status" style="font-size:10px;color:#8aa;margin-top:8px;min-height:12px;"></div>
</div>

<div style="position:fixed;top:20px;right:20px;background:rgba(0,0,0,.9);color:#fff;padding:20px;
            border-left:4px solid #FFFF00;font-family:sans-serif;z-index:9999;">
  <div id="report_count" style="font-size:32px;font-weight:900;">{total_reports}</div>
  <div style="font-size:10px;color:#39FF14;">Illegal Dumping Reports</div>
  <div style="font-size:10px;color:#FFFF00;margin-top:10px;">LIVE TIME</div>
  <div id="live_time" style="font-size:16px;font-weight:700;color:#00D1FF;letter-spacing:1px;">--:--:--</div>
</div>

<div class="legend-drawer" id="legend_drawer">
  <button class="legend-toggle" id="legend_toggle" aria-expanded="false">SHOW DANGER LEGEND</button>
  <div class="legend-panel" id="legend_panel" aria-hidden="true">
    <div class="legend-title">RISK LEVELS</div>
    <div class="legend-row"><span class="legend-dot" style="background:#FF3131"></span>High Risk (81-99%)</div>
    <div class="legend-row"><span class="legend-dot" style="background:#FFFF00"></span>Medium Risk (40-80%)</div>
    <div class="legend-row"><span class="legend-dot" style="background:#39FF14"></span>Low Risk (10-39%)</div>
    <div class="legend-row"><span class="legend-dot" style="background:#00D1FF"></span>User Report</div>
  </div>
</div>

<script src="https://cdn.jsdelivr.net/npm/leaflet@1.9.3/dist/leaflet.js"></script>
<script>
// ── map init ──────────────────────────────────────────────────────────────
const map = L.map('map', {{preferCanvas: true, zoomControl: false}});
map.fitBounds([[{min_lat},{min_lon}],[{max_lat},{max_lon}]]);
L.tileLayer('https://{{s}}.basemaps.cartocdn.com/dark_all/{{z}}/{{x}}/{{y}}{{r}}.png', {{
  attribution: '&copy; OpenStreetMap contributors &copy; CARTO',
  subdomains: 'abcd', maxZoom: 20
}}).addTo(map);

// ── helpers ───────────────────────────────────────────────────────────────
const PORT  = {PORT};
const CACHE = new Map();
const USER_MARKERS = new Map();

function ptColor(r, u) {{
  if (u) return '#00D1FF';
  if (r > 80) return '#FF3131';
  if (r >= 40) return '#FFFF00';
  return '#39FF14';
}}

function esc(s) {{
  return String(s).replace(/[&<>"']/g, c =>
    ({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c]));
}}

function setStatus(msg, color='#8aa') {{
  const sts = document.getElementById('submit_status');
  if (!sts) return;
  sts.style.color = color;
  sts.textContent = msg;
}}

async function fetchJson(url, ms=7000) {{
  const ctrl = new AbortController();
  const t = setTimeout(() => ctrl.abort(), ms);
  try {{
    const r = await fetch(url, {{signal: ctrl.signal}});
    if (!r.ok) throw new Error(r.status);
    return await r.json();
  }} finally {{ clearTimeout(t); }}
}}

async function reverseGeocode(lat, lon) {{
  const key = lat.toFixed(6)+','+lon.toFixed(6);
  if (CACHE.has(key)) return CACHE.get(key);
  try {{
    const d = await fetchJson(
      'https://geocode.arcgis.com/arcgis/rest/services/World/GeocodeServer/reverseGeocode'+
      '?f=pjson&location='+encodeURIComponent(lon+','+lat), 8000);
    const v = d?.address?.ShortLabel || d?.address?.Address || d?.address?.Match_addr
              || lat.toFixed(5)+', '+lon.toFixed(5);
    CACHE.set(key, v);
    return v;
  }} catch {{ return lat.toFixed(5)+', '+lon.toFixed(5); }}
}}

// ── load markers from external JSON ──────────────────────────────────────
function renderMarkers(MARKERS) {{
  MARKERS.forEach(({{"lat":lat,"lon":lon,"r":r,"u":u,"a":a,"id":id}}) => {{
    const color    = ptColor(r, u);
    const label    = u ? 'USER REPORT' : 'RISK: '+r+'%';
    const addrHtml = a
      ? esc(a)
      : '<span class="lazy-addr" data-lat="'+lat+'" data-lon="'+lon+'">Loading\u2026</span>';
    if (u) {{
      addUserMarker(lat, lon, a, id);
      return;
    }}
    L.circleMarker([lat, lon], {{
      radius: u ? 4.8 : 4.2, fillColor: color, fillOpacity: 0.95,
      color, weight: 1.2, opacity: 1,
    }}).addTo(map).bindPopup(
      '<div style="text-align:center;font-family:monospace;">'+
      '<b style="color:'+color+'">'+label+'</b><br>'+
      '<div style="font-size:10px;color:#ccc;margin:8px 0 10px;">'+addrHtml+'</div>'+
      '</div>', {{maxWidth: 240}}
    );
  }});
  document.getElementById('loading').style.display = 'none';
}}

async function removeUserReport(userId, lat, lon, address) {{
  const params = new URLSearchParams();
  if (userId !== null && userId !== undefined) params.set('id', String(userId));
  params.set('lat', String(lat));
  params.set('lon', String(lon));
  if (address) params.set('address', address);
  return fetchJson('http://127.0.0.1:'+PORT+'/remove?'+params.toString(), 9000);
}}

function addUserMarker(lat, lon, address, userId=null) {{
  const markerKey = 'u_' +
    (userId !== null && userId !== undefined ? String(userId) : 'tmp') + '_' +
    lat.toFixed(6) + '_' + lon.toFixed(6) + '_' + Math.random().toString(36).slice(2, 8);
  const safeAddress = address ? esc(address) : 'User-submitted point';
  const marker = L.circleMarker([lat, lon], {{
    radius: 4.8, fillColor: '#00D1FF', fillOpacity: 0.95,
    color: '#00D1FF', weight: 1.2, opacity: 1,
  }}).addTo(map).bindPopup(
    '<div style="text-align:center;font-family:monospace;">'+
    '<b style="color:#00D1FF">USER REPORT</b><br>'+
    '<div style="font-size:10px;color:#ccc;margin:8px 0 10px;">'+safeAddress+'</div>'+
    '<button type="button" class="delete-user-point-btn" '
      + 'data-marker-key="'+markerKey+'" '
      + 'data-user-id="'+(userId !== null && userId !== undefined ? String(userId) : '')+'" '
      + 'data-lat="'+lat+'" '
      + 'data-lon="'+lon+'" '
      + 'data-address="'+safeAddress+'" '
      + 'style="padding:4px 10px;font-size:10px;border:1px solid #ff7676;border-radius:4px;background:#2a1111;color:#ff9e9e;cursor:pointer;">Delete</button>'+
    '</div>', {{maxWidth: 240}}
  );
  USER_MARKERS.set(markerKey, marker);
  return marker;
}}

fetch('./data/{markers_json_file}')
  .then(r => {{ if (!r.ok) throw new Error(r.status); return r.json(); }})
  .then(renderMarkers)
  .catch(err => {{
    console.error('Failed to load markers:', err);
    document.getElementById('loading').textContent = 'Failed to load map data. Is the server running?';
  }});

// lazy-load street address on popup open
map.on('popupopen', async e => {{
  const el = e.popup.getElement()?.querySelector('.lazy-addr');
  if (el && !el.dataset.loaded) {{
    el.dataset.loaded = '1';
    el.textContent = await reverseGeocode(parseFloat(el.dataset.lat), parseFloat(el.dataset.lon));
  }}
  const btn = e.popup.getElement()?.querySelector('.delete-user-point-btn');
  if (btn && !btn.dataset.bound) {{
    btn.dataset.bound = '1';
    btn.addEventListener('click', async ev => {{
      ev.preventDefault();
      ev.stopPropagation();
      const markerKey = btn.dataset.markerKey || '';
      const marker = USER_MARKERS.get(markerKey);
      const userIdText = (btn.dataset.userId || '').trim();
      const userId = userIdText === '' ? null : Number(userIdText);
      const lat = Number(btn.dataset.lat);
      const lon = Number(btn.dataset.lon);
      const address = btn.dataset.address || '';
      if (!confirm('Remove this user-submitted point?')) return;
      try {{
        const d = await removeUserReport(Number.isFinite(userId) ? userId : null, lat, lon, address);
        if (typeof d.total_reports === 'number') publishCount(d.total_reports);
        if (marker) {{
          map.removeLayer(marker);
          USER_MARKERS.delete(markerKey);
        }}
        setStatus('User report removed.', '#39FF14');
      }} catch (err) {{
        console.error(err);
        if (!Number.isFinite(userId)) {{
          const cur = parseInt(document.getElementById('report_count').textContent, 10);
          if (!isNaN(cur) && cur > 0) publishCount(cur - 1);
          if (marker) {{
            map.removeLayer(marker);
            USER_MARKERS.delete(markerKey);
          }}
          setStatus('Removed from this view only (server unavailable).', '#ffaa66');
          return;
        }}
        setStatus('Failed to remove point. Ensure --serve is running.', '#ff7676');
      }}
    }});
  }}
}});

// ── high / low mode toggle ────────────────────────────────────────────────
const HIGH_TOP         = {high_top_json};
const LOW_TOP          = {low_top_json};
const HIGH_LEADERBOARD = {high_leaderboard_js};
const LOW_LEADERBOARD  = {low_leaderboard_js};

const modePanel    = document.getElementById('mode_panel');
const modeTitle    = document.getElementById('mode_title');
const modeList     = document.getElementById('mode_list');
const modeToggle   = document.getElementById('mode_toggle');
const legendDrawer = document.getElementById('legend_drawer');
const legendToggle = document.getElementById('legend_toggle');
const legendPanel  = document.getElementById('legend_panel');
const topLayer     = L.layerGroup().addTo(map);
let   currentMode  = 'high';

legendToggle.addEventListener('click', () => {{
  const isOpen = legendDrawer.classList.toggle('open');
  legendToggle.textContent = isOpen ? 'HIDE DANGER LEGEND' : 'SHOW DANGER LEGEND';
  legendToggle.setAttribute('aria-expanded', String(isOpen));
  legendPanel.setAttribute('aria-hidden', String(!isOpen));
}});

function renderTopPopups(items) {{
  topLayer.clearLayers();
  items.forEach(p => {{
    L.popup({{autoClose:false, closeOnClick:false}})
      .setLatLng([p.lat, p.lon])
      .setContent(
        '<div style="text-align:center;font-family:monospace;">'+
        '<b style="color:'+p.color+'">RISK: '+p.risk+'%</b><br>'+
        '<div style="font-size:10px;color:#ccc;margin-top:6px;">'+p.address+'</div>'+
        '</div>')
      .addTo(topLayer);
  }});
}}

function renderMode(mode) {{
  const isLow = mode === 'low';
  modePanel.style.borderTopColor = isLow ? '#39FF14' : '#FF3131';
  modeTitle.style.color          = isLow ? '#39FF14' : '#FF3131';
  modeTitle.textContent          = isLow ? 'TOP 10 LEAST LIKELY LOCATIONS' : 'TOP 10 RISK LOCATIONS';
  modeList.innerHTML             = isLow ? LOW_LEADERBOARD : HIGH_LEADERBOARD;
  modeToggle.textContent         = isLow ? 'OPEN HIGH-RISK VIEW' : 'OPEN LEAST-LIKELY VIEW';
  renderTopPopups(isLow ? LOW_TOP : HIGH_TOP);
  currentMode = mode;
}}

modeToggle.addEventListener('click', e => {{
  e.preventDefault();
  renderMode(currentMode === 'low' ? 'high' : 'low');
}});
renderMode('high');

// ── report count sync ─────────────────────────────────────────────────────
const COUNT_KEY = 'illegal_dumping_report_count_v1';
const BC = typeof BroadcastChannel !== 'undefined'
  ? new BroadcastChannel('illegal_dumping_report_count_channel_v1') : null;

function setCount(v) {{
  const el = document.getElementById('report_count');
  if (el && Number.isFinite(v)) el.textContent = String(v);
}}
function publishCount(v) {{
  setCount(v);
  try {{ localStorage.setItem(COUNT_KEY, String(v)); }} catch(_) {{}}
  try {{ BC?.postMessage({{type:'report_count',value:v}}); }} catch(_) {{}}
}}
function syncCount() {{
  try {{
    const s = localStorage.getItem(COUNT_KEY);
    if (s !== null) setCount(parseInt(s, 10));
  }} catch(_) {{}}
  window.addEventListener('storage', e => {{
    if (e.key === COUNT_KEY && e.newValue) setCount(parseInt(e.newValue, 10));
  }});
  if (BC) BC.onmessage = e => {{ if (typeof e?.data?.value === 'number') setCount(e.data.value); }};
}}
async function pollCount() {{
  try {{
    const d = await fetchJson('http://127.0.0.1:'+PORT+'/count', 4000);
    if (typeof d.total_reports === 'number') publishCount(d.total_reports);
  }} catch(_) {{}}
}}

// ── live clock ────────────────────────────────────────────────────────────
function tick() {{
  const el = document.getElementById('live_time');
  if (el) el.textContent = new Date().toLocaleTimeString('en-US', {{hour12: false}});
}}

// ── geocode for new report ────────────────────────────────────────────────
async function geocodeAddress(text) {{
  const q   = /oakland/i.test(text) ? text : text + ', Oakland, CA';
  const enc = encodeURIComponent(q);
  try {{
    const d = await fetchJson('http://127.0.0.1:'+PORT+'/geocode?text='+enc, 7000);
    if (typeof d.lat === 'number') return d;
  }} catch(_) {{}}
  try {{
    const d = await fetchJson(
      'https://geocode.arcgis.com/arcgis/rest/services/World/GeocodeServer/findAddressCandidates'+
      '?f=pjson&maxLocations=1&searchExtent='+encodeURIComponent('-122.55,37.68,-122.05,37.92')+'&SingleLine='+enc, 9000);
    const loc = d?.candidates?.[0]?.location;
    if (loc) return {{lat: loc.y, lon: loc.x}};
  }} catch(_) {{}}
  const d = await fetchJson(
    'https://api.geoapify.com/v1/geocode/search?text='+enc+'&bias=proximity:-122.2712,37.8044&apiKey={API_KEY}', 9000);
  const f = d?.features?.[0];
  if (!f) throw new Error('No geocode match');
  return {{lat: f.geometry.coordinates[1], lon: f.geometry.coordinates[0]}};
}}

// ── submit report ─────────────────────────────────────────────────────────
async function submitPlace() {{
  const inp  = document.getElementById('place_in');
  const btn  = document.getElementById('gen_btn');
  const sts  = document.getElementById('submit_status');
  const text = inp.value.trim();
  if (!text) return;
  sts.textContent = '';
  btn.disabled = true;
  btn.textContent = 'CALCULATING\u2026';
  try {{
    const {{lat, lon}} = await geocodeAddress(text);
    let nextCount = null, savedToServer = false, userReportId = null;
    try {{
      const d = await fetchJson(
        'http://127.0.0.1:'+PORT+'/add?lat='+lat+'&lon='+lon+'&address='+encodeURIComponent(text), 9000);
      savedToServer = true;
      if (typeof d.total_reports === 'number') nextCount = d.total_reports;
      if (typeof d.user_report_id === 'number') userReportId = d.user_report_id;
    }} catch(_) {{
      sts.textContent = 'Saved locally in this view. Start --serve to persist.';
    }}
    if (nextCount === null) {{
      const cur = parseInt(document.getElementById('report_count').textContent, 10);
      if (!isNaN(cur)) nextCount = cur + 1;
    }}
    if (nextCount !== null) publishCount(nextCount);
    addUserMarker(lat, lon, text, userReportId);
    inp.value = '';
    btn.textContent = savedToServer ? 'SENT' : 'SENT (LOCAL)';
  }} catch(err) {{
    console.error(err);
    sts.textContent = 'Address not found. Try full street + Oakland, CA.';
    btn.textContent = 'TRY AGAIN';
  }} finally {{
    setTimeout(() => {{ btn.disabled = false; btn.textContent = 'Enter'; }}, 1400);
  }}
}}

// ── boot ──────────────────────────────────────────────────────────────────
syncCount();
tick(); setInterval(tick, 1000);
pollCount(); setInterval(pollCount, 2500);
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Main build function
# ---------------------------------------------------------------------------

def build_map():
    global latest_total_reports
    session = None
    try:
        PUBLIC_DIR.mkdir(parents=True, exist_ok=True)

        if not is_rebuild_needed():
            print("   markers.json is up to date — skipping rebuild.")
            try:
                count_path = PUBLIC_DIR / "data" / "total_reports.json"
                with open(count_path) as f:
                    total = json.load(f)["total"]
                with report_count_lock:
                    latest_total_reports = total
                print(f"   Report count loaded: {total:,}")
            except Exception as e:
                print(f"   Warning: could not read total_reports.json: {e}")
            return latest_total_reports

        session = create_retry_session()
        base_url = f"https://data.oaklandca.gov/resource/{OAKLAND_DATA_ID}.json"

        print("1. Loading city data...")
        city_rows = load_city_cache()
        if city_rows is None:
            print("   Fetching from Oakland API...")
            city_rows = fetch_all_city_rows(session, base_url)
            save_city_cache(city_rows)
            print(f"   Fetched {len(city_rows):,} rows and saved to cache.")
        city_df = pd.DataFrame(city_rows)
        city_df = city_df.rename(columns={'srx': 'longitude', 'sry': 'latitude'})
        if 'address' not in city_df.columns:
            city_df['address'] = ''
        city_df['address'] = city_df['address'].fillna('').astype(str).str.strip()
        city_df[['latitude','longitude']] = city_df[['latitude','longitude']].apply(pd.to_numeric, errors='coerce')
        city_df = city_df.dropna(subset=['latitude','longitude'])
        city_df = filter_to_oakland_boundary(city_df)
        city_df['source'] = 'city'

        with open(LOCAL_DB) as f:
            local_data = json.load(f)
        for idx, row in enumerate(local_data):
          if isinstance(row, dict):
            row["rid"] = idx
        local_df = pd.DataFrame(local_data) if local_data \
                   else pd.DataFrame(columns=['latitude','longitude'])
        if 'address' not in local_df.columns:
            local_df['address'] = ''
        if 'rid' not in local_df.columns:
          local_df['rid'] = pd.Series(dtype='Int64')
        local_df['address'] = local_df['address'].fillna('').astype(str).str.strip()
        local_df['source']  = 'user'
        local_df['rid'] = pd.to_numeric(local_df['rid'], errors='coerce').astype('Int64')
        local_df[['latitude','longitude']] = local_df[['latitude','longitude']].apply(pd.to_numeric, errors='coerce')
        local_df = local_df.dropna(subset=['latitude','longitude'])
        local_df = filter_to_oakland_boundary(local_df)

        city_df['rid'] = pd.NA

        df = pd.concat([city_df, local_df], ignore_index=True)
        df['address'] = df['address'].fillna('').astype(str).str.strip()
        df = assign_density_risk_scores(df)
        total_reports = len(city_df) + len(local_df)
        with report_count_lock:
            latest_total_reports = total_reports

        def top_n(asc, n=10):
            return (df.sort_values(
                        ["risk_pct","local_density","latitude","longitude"],
                        ascending=[asc, asc, True, True])
                      .drop_duplicates(subset=["_rlat","_rlon"])
                      .head(n))

        top_high = top_n(asc=False)
        top_low  = top_n(asc=True)

        if len(df) > MAX_RENDER_POINTS:
            must_idx  = set(top_high.index) | set(top_low.index) | set(df[df['source']=='user'].index)
            must_df   = df.loc[sorted(must_idx)]
            remaining = df.drop(index=must_idx)
            slots     = max(0, MAX_RENDER_POINTS - len(must_df))
            sampled   = spread_sample(remaining, slots, "latitude", "longitude",
                                      RENDER_GRID_SIZE, RENDER_SAMPLE_SEED, "local_density") \
                        if slots > 0 else remaining.iloc[0:0]
            render_df = pd.concat([must_df, sampled], ignore_index=True)
        else:
            render_df = df.copy()

        render_df["latitude"]  = render_df["latitude"].round(6)
        render_df["longitude"] = render_df["longitude"].round(6)
        bounds = get_point_bounds(render_df)

        markers = [
            {
                "lat": round(float(r.latitude),  6),
                "lon": round(float(r.longitude), 6),
                "r":   int(r.risk_pct),
                "u":   1 if r.source == "user" else 0,
                "a":   str(r.address).strip(),
                "id": int(r.rid) if r.source == "user" and pd.notna(r.rid) else None,
            }
            for r in render_df.itertuples(index=False)
        ]
        markers_file = "markers.json"
        markers_path = PUBLIC_DIR / "data" / markers_file
        markers_path.parent.mkdir(parents=True, exist_ok=True)
        markers_path.write_text(json.dumps(markers, separators=(",", ":")), encoding="utf-8")
        print(f"Wrote {markers_path}  ({markers_path.stat().st_size:,} bytes)")

        count_path = PUBLIC_DIR / "data" / "total_reports.json"
        count_path.write_text(json.dumps({"total": total_reports}), encoding="utf-8")

        html_str = render_map_html(
            top_high_rows    = top_high,
            top_low_rows     = top_low,
            total_reports    = total_reports,
            map_bounds       = bounds,
            markers_json_file= markers_file,
        )
        DASHBOARD_FILE.write_text(html_str, encoding="utf-8")
        print(f"Wrote {DASHBOARD_FILE}  ({len(html_str):,} chars)")

        return total_reports

    except Exception as e:
        print(f"Error building map: {e}")
        return None
    finally:
        if session:
            session.close()


# ---------------------------------------------------------------------------
# HTTP server
# ---------------------------------------------------------------------------

CONTENT_TYPES = {
    '.html': 'text/html',
    '.json': 'application/json',
    '.js':   'application/javascript',
    '.css':  'text/css',
}

class RequestHandler(BaseHTTPRequestHandler):
    def _json(self, code, payload):
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _file(self, path: Path):
        body  = path.read_bytes()
        ctype = CONTENT_TYPES.get(path.suffix.lower(), 'application/octet-stream')
        self.send_response(200)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        global latest_total_reports
        parsed = urllib.parse.urlparse(self.path)
        query  = urllib.parse.parse_qs(parsed.query)
        p      = parsed.path

        if p == '/count':
            with report_count_lock:
                self._json(200, {"ok": True, "total_reports": latest_total_reports})

        elif p == '/geocode':
            text = query.get('text', [''])[0].strip()
            if not text:
                return self._json(400, {"ok": False, "error": "Missing text"})
            try:
                r = requests.get(
                    "https://geocode.arcgis.com/arcgis/rest/services/World/GeocodeServer/findAddressCandidates",
                    params={"SingleLine": text, "f": "pjson", "maxLocations": 1,
                            "searchExtent": "-122.55,37.68,-122.05,37.92"},
                    timeout=8,
                )
                r.raise_for_status()
                cands = r.json().get("candidates", [])
                if not cands:
                    return self._json(404, {"ok": False, "error": "No match"})
                loc = cands[0]["location"]
                self._json(200, {"ok": True, "lat": float(loc["y"]), "lon": float(loc["x"])})
            except Exception as e:
                self._json(500, {"ok": False, "error": str(e)})

        elif p == '/add':
            try:
                lat  = float(query['lat'][0])
                lon  = float(query['lon'][0])
                addr = query.get('address', [''])[0].strip()[:160]
            except (KeyError, ValueError):
                return self._json(400, {"ok": False, "error": "Missing/invalid lat or lon"})
            if not is_in_oakland_boundary(lat, lon):
                return self._json(400, {"ok": False, "error": "Outside Oakland bounds"})
            with open(LOCAL_DB, 'r+') as f:
                data = json.load(f)
                data.append({"latitude": lat, "longitude": lon, "source": "user", "address": addr})
                user_report_id = len(data) - 1
                f.seek(0); json.dump(data, f); f.truncate()
            with report_count_lock:
                latest_total_reports += 1
                total = latest_total_reports
            rebuild_map_async()
            self._json(200, {"ok": True, "added": 1, "total_reports": total, "user_report_id": user_report_id})

        elif p == '/remove':
            rid_raw = query.get('id', [''])[0].strip()
            lat_raw = query.get('lat', [''])[0].strip()
            lon_raw = query.get('lon', [''])[0].strip()
            addr = query.get('address', [''])[0].strip()
            rid = None
            lat = None
            lon = None
            if rid_raw:
                try:
                    rid = int(rid_raw)
                except ValueError:
                    return self._json(400, {"ok": False, "error": "Invalid id"})
            if lat_raw and lon_raw:
                try:
                    lat = float(lat_raw)
                    lon = float(lon_raw)
                except ValueError:
                    return self._json(400, {"ok": False, "error": "Invalid lat/lon"})

            with open(LOCAL_DB, 'r+') as f:
                data = json.load(f)
                remove_idx = None

                if rid is not None and 0 <= rid < len(data):
                    remove_idx = rid

                if remove_idx is None and lat is not None and lon is not None:
                    for i, row in enumerate(data):
                        try:
                            rlat = float(row.get("latitude"))
                            rlon = float(row.get("longitude"))
                        except (TypeError, ValueError):
                            continue
                        if abs(rlat - lat) > 1e-6 or abs(rlon - lon) > 1e-6:
                            continue
                        row_addr = str(row.get("address", "")).strip()
                        if addr and row_addr and row_addr != addr:
                            continue
                        remove_idx = i
                        break

                if remove_idx is None:
                    return self._json(404, {"ok": False, "error": "User report not found"})

                data.pop(remove_idx)
                f.seek(0); json.dump(data, f); f.truncate()

            with report_count_lock:
                latest_total_reports = max(0, latest_total_reports - 1)
                total = latest_total_reports
            rebuild_map_async()
            self._json(200, {"ok": True, "removed": 1, "total_reports": total})

        elif p == '/' or p == '':
            self._file(DASHBOARD_FILE)

        elif p.startswith('/public/'):
            file_path = PUBLIC_DIR / p[len('/public/'):]
            if file_path.exists() and file_path.is_file():
                self._file(file_path)
            else:
                self._json(404, {"ok": False, "error": "Not found"})

        elif p.startswith('/map/public/'):
          file_path = PUBLIC_DIR / p[len('/map/public/'):]
          if file_path.exists() and file_path.is_file():
            self._file(file_path)
          else:
            self._json(404, {"ok": False, "error": "Not found"})

        elif p == '/map' or p == '/map/':
          self._file(DASHBOARD_FILE)

        else:
            self._json(404, {"ok": False, "error": "Not found"})

    def log_message(self, *_):
        pass


def rebuild_map_async():
    if rebuild_lock.locked():
        return
    def _worker():
        with rebuild_lock:
            build_map()
    threading.Thread(target=_worker, daemon=True).start()


def start_periodic_rebuild(interval_seconds: int, stop_event: threading.Event):
    def _loop():
        while not stop_event.wait(interval_seconds):
            rebuild_map_async()
    threading.Thread(target=_loop, daemon=True).start()


def run_server(open_map=False):
    try:
        srv = HTTPServer(("127.0.0.1", PORT), RequestHandler)
    except OSError as e:
        # Unix: 48/98, Windows: 10048
        if e.errno in (48, 98, 10048) or getattr(e, "winerror", None) == 10048:
            print(f"Port {PORT} already in use; skipped server startup.")
            return
        raise
    print(f"Serving at http://127.0.0.1:{PORT}/public/{DASHBOARD_FILE_NAME}")
    print(
        "Periodic refresh enabled: "
        f"every {CACHE_REFRESH_INTERVAL_SECONDS}s "
        f"(cache max age: {CACHE_MAX_AGE_HRS}h)."
    )
    stop_event = threading.Event()
    start_periodic_rebuild(CACHE_REFRESH_INTERVAL_SECONDS, stop_event)
    if open_map:
        import webbrowser
        webbrowser.open(f"http://127.0.0.1:{PORT}/public/{DASHBOARD_FILE_NAME}")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("Server stopped.")
    finally:
        stop_event.set()
        srv.server_close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--open",  action="store_true")
    args = parser.parse_args()
    try:
        build_map()
    except KeyboardInterrupt:
        raise SystemExit(130)
    if args.serve:
        run_server(open_map=args.open)
    elif args.open:
        import webbrowser
        webbrowser.open(f"http://127.0.0.1:{PORT}/public/{DASHBOARD_FILE_NAME}")