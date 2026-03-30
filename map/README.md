# Live Map & Real-Time Update Center

This project visualizes illegal-dumping activity across Oakland using the city's 311 dataset and locally submitted field reports. It generates interactive HTML maps that show citywide activity, highlight likely hotspots, and keep report totals synchronized across multiple views.

## What the Feature Does

- Pulls Oakland illegal-dumping report coordinates from the public 311 dataset.
- Cleans and normalizes the latitude and longitude values before mapping.
- Renders interactive red, yellow, and green dots on a dark basemap.
- Shows a street-level location when a user clicks a dot.
- Generates separate high-risk and low-risk dashboard pages with top-10 summaries.
- Accepts new user-entered report locations and updates the total report count in real time.

## How It Was Built

### 1. Data ingestion

The feature starts by pulling records from Oakland's Socrata endpoint:

- Dataset ID: `dmqt-4g4v`
- Transport: `requests`
- Pagination: handled in batches so the app can cover the full city dataset
- Reliability: retry logic is used to recover from temporary network or rate-limit failures

In [`src/build_city_map.py`](src/build_city_map.py), the script fetches rows in chunks, filters them to Oakland bounding coordinates, and prepares them for the standalone map output.

In [`src/build_live_dashboard.py`](src/build_live_dashboard.py), the app fetches the city dataset and then merges it with locally saved user reports from [`data/user_reports.json`](data/user_reports.json).

### 2. Data cleaning and normalization

After the raw rows are downloaded, the pipeline:

- renames `srx` and `sry` into longitude and latitude fields
- converts coordinates to numeric values with `pandas`
- drops invalid rows with missing coordinates
- trims and standardizes address values where available

This step makes the mapping logic consistent even when source records are incomplete or formatted inconsistently.

### 3. Likelihood and risk visualization

The project contains two main visualization paths:

- [`src/build_city_map.py`](src/build_city_map.py) uses a density-based approach. It bins reports into a grid, measures neighborhood density across adjacent cells, and converts the percentile rank into three levels:
  - low likelihood: green
  - medium likelihood: yellow
  - high likelihood: red
- [`src/build_live_dashboard.py`](src/build_live_dashboard.py) builds the dashboard experience, merges city and user reports, assigns a `risk_pct` used for color, ranking, and top-10 summaries, and then generates both the high-risk and low-risk pages.

This lets the project support both a straightforward map view and a more operational dashboard view.

### 4. Map rendering

The maps are built with `folium`, which outputs Leaflet-based HTML pages. This approach was chosen so the final artifact can be opened directly in a browser without requiring a separate frontend framework.

Each report is rendered as a `CircleMarker` with:

- a color indicating likelihood or risk
- a popup showing the label and location
- a dark basemap for high contrast

The markers are baked directly into the generated HTML, which makes the output portable and avoids the blank-map problem that can happen when too much rendering is pushed into late client-side JavaScript.

### 5. Location popups

When a popup opens, the client attempts to reverse-geocode the selected coordinate into a readable street label. This is done in browser-side JavaScript after the HTML map loads.

The current implementation uses external geocoding services to:

- turn user-entered addresses into coordinates
- turn clicked coordinates back into street names for popups

An in-memory cache is used to avoid repeating the same reverse-geocode request for the same point.

### 6. Real-time update flow

The "live" behavior is handled by the local server inside [`src/build_live_dashboard.py`](src/build_live_dashboard.py).

When run with `--serve`, it starts a lightweight `HTTPServer` on `127.0.0.1:8080` with these endpoints:

- `/geocode`
  - converts a typed address into latitude and longitude
- `/add`
  - saves a new report into [`data/user_reports.json`](data/user_reports.json)
  - increments the total report count
  - triggers an asynchronous map rebuild
- `/count`
  - returns the latest total report count so multiple pages can stay synchronized

On the frontend, the generated pages:

- submit new locations through JavaScript
- add the new point to the visible map immediately
- synchronize counts with `localStorage`
- synchronize counts across open tabs using `BroadcastChannel`
- poll the local `/count` endpoint periodically to keep both pages aligned
- update the live time display every second

This is what enables the "real-time update center" behavior in the current local deployment model.

## Output Files

The main generated outputs are:

- [`public/oakland_city_map.html`](public/oakland_city_map.html)
  - standalone citywide map view
- [`public/oakland_dashboard.html`](public/oakland_dashboard.html)
  - unified dashboard with high-risk and low-risk modes (`#high` / `#low`)

The repo is organized so the active Python code lives in `src/`, generated site assets live in `public/`, and runtime JSON data lives in `data/`.

## Main Files

- [`src/build_live_dashboard.py`](src/build_live_dashboard.py)
  - primary generator for the dashboard pages and local live-update server
- [`src/build_city_map.py`](src/build_city_map.py)
  - standalone full-map generator using density-based likelihood scoring
- [`data/user_reports.json`](data/user_reports.json)
  - locally stored user-submitted reports
- `public/`
  - website-ready HTML map outputs

## Tech Stack

- Python
- Pandas
- Requests
- Folium / Leaflet
- NumPy
- Browser-side JavaScript
- Lightweight built-in HTTP server

## How to Run It

Install the Python packages used by the project:

```bash
pip install pandas requests folium numpy
```

Generate the standalone map:

```bash
python3 src/build_city_map.py
```

Generate the dashboard pages:

```bash
python3 src/build_live_dashboard.py
```

Run the dashboard with live local endpoints:

```bash
python3 src/build_live_dashboard.py --serve --open
```

If you want to serve the generated HTML files through a simple local web server:

```bash
python3 -m http.server --directory public 8000
```

Then open:

- `http://127.0.0.1:8000/oakland_city_map.html`
- `http://127.0.0.1:8000/oakland_dashboard.html`
- `http://127.0.0.1:8000/oakland_dashboard.html#low`

## Performance Notes

- The project can fetch the full 311 coordinate dataset, but the number of rendered markers is capped for browser performance.
- `MAX_RENDER_POINTS` is used to keep the generated HTML responsive.
- The dashboard preserves top-ranked locations and user-submitted points even when sampling is used for the rest of the map.

## Current Deployment Notes

- The current implementation is best suited for local demo and prototype use.
- The geocoding key in the code should be moved to environment variables before production deployment.
- The real-time behavior is local-machine real time, not a cloud-hosted multi-user backend.

## Summary

This feature was built as a Python-first geospatial dashboard pipeline: public Oakland 311 data is fetched, cleaned, scored, and rendered into shareable interactive maps, while a lightweight local server adds live address submission, synchronized report totals, and cross-page updating for a more operational city response experience.

