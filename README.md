Pranav's Role:
As the Technical Lead and product manager for this feature, I owned the illegal dumping mapping system end-to-end, from defining requirements to shipping a working product. I scoped the architecture, made the core technical decisions, and built the entire pipeline myself: connecting to Oakland's Socrata dataset via a City of Oakland public API key, cleaning and validating coordinates, engineering a GeoJSON boundary module with Shapely geometry checks and custom exclusion zones for geographic accuracy, and designing a grid-based density scoring system that assigns risk percentages and drives the green, yellow, and red color coding across every view. On the frontend, I used Folium and Leaflet to build interactive HTML maps with clickable popups and readable street names, and on the backend, I built a local server handling geocoding, report submission, JSON persistence, Oakland boundary validation, and real-time synchronization across all three views: the citywide map, the high-risk dashboard, and the low-risk dashboard with a toggle between them. Beyond my own feature, I regularly checked in with teammates to track progress and surface blockers, and communicated directly with our PI, Erica Astrella, to align on project goals and deliverables. Every decision, from smart sampling for browser performance to retry logic for API reliability, was made with scalability and real-world usability in mind.

# OaklandDumpingFrontend
Frontend dashboard for Oakland illegal dumping detection system

## Project structure

```
src/
	assets/
		cameras.json
	pages/
		index.html
		processed-static.html
		processed-dynamic.html
	scripts/
		script.js
		media-object.js
	styles/
		styles.css
```

Open `src/pages/index.html` to run the frontend locally.

## Frontend Static Server

For frontend-only development, you can run:

```bash
npx http-server src -o pages/index.html -c-1
```

What this does:

- Serves the `src` directory over HTTP
- Automatically opens `pages/index.html`
- Disables browser caching (`-c-1`) so you always see the latest changes

Note: this command only serves static frontend files. It does not start the backend API or the map live service.

## Run All Local Services

Start the detection backend and map dashboard together:

```bash
python run_all.py
```

This starts:

- Backend API at `http://127.0.0.1:8000`
- Map dashboard at `http://127.0.0.1:8080/public/oakland_dashboard.html`

Press `Ctrl+C` to stop both services.

Note: Both commands must be run simultaneously for this project to run as inteded. 
