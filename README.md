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