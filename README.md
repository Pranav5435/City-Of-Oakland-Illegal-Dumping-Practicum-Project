# Roles:

## Pranav:
As the Technical Lead and product manager for this feature, I owned the illegal dumping mapping system end-to-end, from defining requirements to shipping a working product. I scoped the architecture, made the core technical decisions, and built the entire pipeline myself: connecting to Oakland's Socrata dataset via a City of Oakland public API key, cleaning and validating coordinates, engineering a GeoJSON boundary module with Shapely geometry checks and custom exclusion zones for geographic accuracy, and designing a grid-based density scoring system that assigns risk percentages and drives the green, yellow, and red color coding across every view. On the frontend, I used Folium and Leaflet to build interactive HTML maps with clickable popups and readable street names, and on the backend, I built a local server handling geocoding, report submission, JSON persistence, Oakland boundary validation, and real-time synchronization across all three views: the citywide map, the high-risk dashboard, and the low-risk dashboard with a toggle between them. Beyond my own feature, I regularly checked in with teammates to track progress and surface blockers, and communicated directly with our PI, Erica Astrella, to align on project goals and deliverables. Every decision, from smart sampling for browser performance to retry logic for API reliability, was made with scalability and real-world usability in mind.

## Alexander:

As the person who was mostly in charge of gather images and labeling images, I have gathered ~100 images off the web while also going through data sets to check to see if images are compatable with our specifications. After gathering images that would meet our training demands, I then used the free website roboflow to label and export these images. Images were labeled both very specifically and very generally. I would label an image as "Plastic", "Cardboard", "Metal", etc then I would label it as just "Trash". 


## Naya:

## Henry:

## Haoyang:

* Designed the dynamic process of detecting illegal dumpings with 3 iterations, the code is on [colab](https://colab.research.google.com/drive/1yD4aRgR36VuThiYNIqRP35prEBQF5KQ1#scrollTo=YsMzcQIOBJr9)

```
My first version detects human skeletons and tries to see if hands connect to anything. However, I abandoned this method after some testing. First of all, because it's from the camera's angle, the skeletons are not that clear. Secondly, it requires someone to hold the trash continuously for 3 seconds to avoid false positives, which is not very practical.

In this case, I switched to the current method:

Two foreground extractors named A and B. I use the CNT foreground extractor. A has a fast learning rate, and B has a slow learning rate, which means that A can absorb new objects faster and B is slower.

When a person enters the area, YOLO will assign that person an ID, and after that ID has left the area for a certain number of seconds, we can say that that person has disappeared. However, the YOLO tracking algorithm is not as good as I thought, so I also use CNT A to enhance this tracking process.

Once the person has disappeared, we can look through their path and see if there is any change, as CNT B has a slow learning rate and new objects have not been absorbed yet.

If there is a change, there are two situations:

    A new object
    A disappeared object


To handle this, we first use YOLO to "whitelist" some common objects, like cars and bicycles. After that, I use contour extraction to see if the object has very complex contours, as trash usually has much more complex contours than the background. I also use color comparison to see if that area has a very unusual color, as trash usually comes in all different kinds of colors.

If the confidence is higher than 0.6, the program will trigger the alert.

This process can reach at least 4.5 frames per second, and as fast as 6 frames per second depends on the number of people in the scene.
```

* Designed an API backend which can be used in the future: See [1234567Yang/IllegalDumpingAPIBackend](https://github.com/1234567Yang/IllegalDumpingAPIBackend)
* Designed the database structure with 1-3NF: See [database/](database/)


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
