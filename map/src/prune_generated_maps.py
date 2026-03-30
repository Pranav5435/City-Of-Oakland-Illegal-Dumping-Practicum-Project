import json
import re
from pathlib import Path

from oakland_boundary_utils import is_in_oakland_boundary
from shapely.geometry import Point, Polygon


SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent if SCRIPT_DIR.name == "src" else SCRIPT_DIR
PUBLIC_FILES = (
    BASE_DIR / "public" / "oakland_city_map.html",
    BASE_DIR / "public" / "oakland_dashboard.html",
)

MARKER_BLOCK_RE = re.compile(
    r'var circle_marker_(?P<marker>[a-f0-9]+) = L\.circleMarker\('
    r'\s*\[(?P<lat>-?\d+(?:\.\d+)?), (?P<lon>-?\d+(?:\.\d+)?)\],'
    r'.*?'
    r'circle_marker_(?P=marker)\.bindPopup\(popup_[a-f0-9]+\)\s*;'
    r'\s*',
    re.S,
)
TOP_POINTS_RE = re.compile(r"const topPoints = (\[.*?\]);", re.S)
MARKER_POINT_RE = re.compile(
    r'L\.circleMarker\(\s*\[(?P<lat>-?\d+(?:\.\d+)?), (?P<lon>-?\d+(?:\.\d+)?)\],'
    r'\s*(?P<options>\{.*?\})',
    re.S,
)
MARKER_COLOR_RE = re.compile(r'"(?:fillColor|color)": "(#[A-F0-9]+)"')
BACKFILL_SCRIPT_RE = re.compile(
    r"\s*<!-- OAKLAND_BACKFILL_START -->.*?<!-- OAKLAND_BACKFILL_END -->\s*",
    re.S,
)
BACKFILL_ZONES = (
    {
        "polygon": Polygon(
            [
                (-122.2700, 37.7910),
                (-122.2680, 37.7880),
                (-122.2620, 37.7760),
                (-122.2460, 37.7760),
                (-122.2460, 37.7910),
            ]
        ),
        "bounds": (37.7760, 37.7910, -122.2700, -122.2460),
        "step": 0.0040,
        "occupied_window": 0.0035,
        "neighbor_window": 0.0200,
        "min_neighbors": 1000,
    },
)
COLOR_TO_RISK = {
    "#39FF14": 10,
    "#FFFF00": 60,
    "#FF3131": 95,
    "#00D1FF": 1,
}


def should_keep(lat: float, lon: float) -> bool:
    return is_in_oakland_boundary(float(lat), float(lon))


def prune_marker_blocks(text: str):
    removed = []

    def replace(match: re.Match[str]) -> str:
        lat = float(match.group("lat"))
        lon = float(match.group("lon"))
        if should_keep(lat, lon):
            return match.group(0)
        removed.append((lat, lon))
        return ""

    return MARKER_BLOCK_RE.sub(replace, text), removed


def prune_top_points(text: str):
    removed = []

    def replace(match: re.Match[str]) -> str:
        points = json.loads(match.group(1))
        kept = []
        for point in points:
            lat = float(point["lat"])
            lon = float(point["lon"])
            if should_keep(lat, lon):
                kept.append(point)
            else:
                removed.append((lat, lon))
        return f"const topPoints = {json.dumps(kept)};"

    return TOP_POINTS_RE.sub(replace, text), removed


def extract_marker_points(text: str):
    points = []
    for match in MARKER_POINT_RE.finditer(text):
        color_match = MARKER_COLOR_RE.search(match.group("options"))
        if not color_match:
            continue
        points.append(
            {
                "lat": float(match.group("lat")),
                "lon": float(match.group("lon")),
                "color": color_match.group(1),
            }
        )
    return points


def get_backfill_points(text: str):
    markers = extract_marker_points(text)
    if not markers:
        return []

    backfill = []
    for zone in BACKFILL_ZONES:
        polygon = zone["polygon"]
        min_lat, max_lat, min_lon, max_lon = zone["bounds"]
        lat = min_lat
        while lat <= max_lat + 1e-9:
            lon = min_lon
            while lon <= max_lon + 1e-9:
                point = Point(lon, lat)
                if polygon.covers(point) and should_keep(lat, lon):
                    occupied = any(
                        abs(marker["lat"] - lat) <= zone["occupied_window"]
                        and abs(marker["lon"] - lon) <= zone["occupied_window"]
                        for marker in markers
                    )
                    if not occupied:
                        nearby = [
                            marker
                            for marker in markers
                            if abs(marker["lat"] - lat) <= zone["neighbor_window"]
                            and abs(marker["lon"] - lon) <= zone["neighbor_window"]
                        ]
                        if len(nearby) >= zone["min_neighbors"]:
                            nearest = min(
                                nearby,
                                key=lambda marker: (marker["lat"] - lat) ** 2 + (marker["lon"] - lon) ** 2,
                            )
                            color = nearest["color"]
                            backfill.append(
                                {
                                    "lat": round(lat, 6),
                                    "lon": round(lon, 6),
                                    "color": color,
                                    "risk": COLOR_TO_RISK.get(color, 50),
                                }
                            )
                lon += zone["step"]
            lat += zone["step"]
    return backfill


def render_backfill_script(path: Path, points_to_add):
    if not points_to_add:
        return ""

    points_json = json.dumps(points_to_add, separators=(",", ":"))
    if path.name == "oakland_city_map.html":
        popup_builder = """
        const label = p.color === "#FF3131" ? "HIGH" : (p.color === "#FFFF00" ? "MEDIUM" : "LOW");
        const popupHtml =
            '<div class="popup-card">' +
            '<div class="popup-row"><b>Street:</b> <span class="street-address" data-lat="' + p.lat + '" data-lon="' + p.lon + '">Loading...</span></div>' +
            '<div><b>Likelihood:</b> ' + label + '</div>' +
            '</div>';
        L.circleMarker([p.lat, p.lon], {
            radius: 7,
            fill: true,
            fillColor: p.color,
            fillOpacity: 1.0,
            color: null,
            weight: 0,
        }).addTo(mapObj).bindPopup(popupHtml, { maxWidth: 280 });
        """
    else:
        popup_builder = """
        const popupHtml =
            '<div class="popup-card"><b style="color:' + p.color + '">RISK: ' + p.risk + '%</b>' +
            '<div class="popup-loc"><span class="popup-location" data-lat="' + p.lat + '" data-lon="' + p.lon + '">Loading...</span></div>' +
            '</div>';
        L.circleMarker([p.lat, p.lon], {
            bubblingMouseEvents: true,
            color: p.color,
            fill: true,
            fillColor: p.color,
            fillOpacity: 0.92,
            fillRule: "evenodd",
            lineCap: "round",
            lineJoin: "round",
            opacity: 1.0,
            radius: 4.0,
            stroke: true,
            weight: 1.0,
        }).addTo(mapObj).bindPopup(popupHtml, { maxWidth: 240 });
        """

    return f"""
<!-- OAKLAND_BACKFILL_START -->
<script>
(function() {{
    const backfillPoints = {points_json};
    function findMap() {{
        if (typeof L === "undefined") return null;
        for (const value of Object.values(window)) {{
            if (value instanceof L.Map) return value;
        }}
        return null;
    }}
    function addBackfill() {{
        const mapObj = findMap();
        if (!mapObj) {{
            setTimeout(addBackfill, 150);
            return;
        }}
        if (mapObj._oaklandBackfillApplied) return;
        mapObj._oaklandBackfillApplied = true;
        backfillPoints.forEach((p) => {{
            {popup_builder}
        }});
    }}
    window.addEventListener("load", addBackfill);
}})();
</script>
<!-- OAKLAND_BACKFILL_END -->
"""


def inject_backfill(path: Path, text: str):
    script = render_backfill_script(path, get_backfill_points(text))
    if not script:
        return text, 0
    if "</body>" not in text:
        return text + script, script.count('"lat"')
    return text.replace("</body>", script + "\n</body>"), script.count('"lat"')


def prune_file(path: Path):
    original = path.read_text(encoding="utf-8")
    text = BACKFILL_SCRIPT_RE.sub("", original)
    text, removed_markers = prune_marker_blocks(text)
    text, removed_top_points = prune_top_points(text)
    text, added_backfill = inject_backfill(path, text)
    if text != original:
        path.write_text(text, encoding="utf-8")
    return removed_markers, removed_top_points, added_backfill


def main():
    for path in PUBLIC_FILES:
        if not path.exists():
            print(f"{path.name}: skipped (file not found)")
            continue
        removed_markers, removed_top_points, added_backfill = prune_file(path)
        print(
            f"{path.name}: removed {len(removed_markers)} marker blocks, "
            f"{len(removed_top_points)} topPoints entries, "
            f"added {added_backfill} backfill markers"
        )


if __name__ == "__main__":
    main()
