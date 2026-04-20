# Roles:

## Pranav:
As the Technical Lead, I owned the illegal dumping mapping system from start to finish. That meant scoping the architecture, making the core technical decisions, and building the entire pipeline myself. The standalone version of this feature before integration lives at https://github.com/Pranav5435/Live-Map-Real-Time-Update-Center. On the data side, I connected to Oakland's Socrata dataset using a City of Oakland public API key and pulled in 49,992 real 311 service requests. From there I cleaned and validated every coordinate before it touched the model. I built a GeoJSON boundary module using Shapely that checks whether each report actually falls inside Oakland city limits, and I added custom exclusion zones to filter out noise from the bay waterline and areas with no real street frontage. Without that step the map would have had bogus risk scores in places that are not actually dumpable locations. For the prediction model I used XGBoost, which I chose because it handles nonlinear feature interactions naturally and produces probability outputs that are easy to convert into a clean risk score. I indexed all the validated coordinates into Uber's H3 hexagonal grid at resolution 9, which gives cells about 105 meters across. Each cell gets a density score based on historical incident counts across rolling time windows, and then KDE smoothing is applied over the cell centroids to produce a continuous risk surface rather than a choppy histogram. The final output assigns every cell a risk percentage from 0 to 100, and those scores drive the green, yellow, and red color coding across all three map views. On the frontend I used Folium and Leaflet to build interactive HTML maps with a CartoDB dark tile basemap. Every hexagonal cell is clickable and shows the risk score, incident count, and nearest street address from reverse geocoding. One problem I ran into was performance. Rendering all 15,000 populated cells in Folium was crashing browsers, so I built a smart sampling system that keeps every red cell intact, keeps most of the yellow cells, and downsamples the green cells significantly. That brought the rendered count down to around 3,200 cells while preserving everything that actually matters operationally. The backend is a local FastAPI server that handles geocoding, incoming report submissions with JSON persistence, Oakland boundary validation on submitted coordinates, and real time synchronization across all three dashboard views on a 30 second refresh cycle. Beyond my own feature I regularly checked in with teammates to surface blockers early and communicated directly with our PI Erica Astrella to keep the project aligned with what the city actually needed.

## Alexander:

As the person who was mostly in charge of gather images and labeling images, I have gathered ~100 images off the web while also going through data sets to check to see if images are compatable with our specifications. After gathering images that would meet our training demands, I then used the free website roboflow to label and export these images. Images were labeled both very specifically and very generally. I would label an image as "Plastic", "Cardboard", "Metal", etc then I would label it as just "Trash". 


## Naya:

Contributed to dataset development through image collection and bounding box annotation for training the detection models. Designed and built the project's frontend interface using HTML, CSS, and JavaScript — serving as the primary demo and showcase layer for the pipeline. Responsible for integrating the two core models into a unified frontend experience, using JSON serialization to handle data exchange between model outputs and the UI.

## Henry:

Designed the static process of detecting illegal dumping within stationary images. I accomplished this by using roboflow and uloading images from our dataset to train the pre-exiting roboflow model (RF-DETR). The project then uses Roboflow's serverless API for the model I trained to create trash detection boxes within images. At this time Roboflow did not support the ability to download the model we trained, using an API was the only method we had to detect trash within images. 

The project runs two processes simultaneously for the backend: the image detection processes and the map hosting process. Both are run automatically with the command `python run_all.py`. This lets the frontend communicate with the backend by making a request from a local sever, because it can not do so directly from the frontend because of CORS.This also emulates how it would work if the frontend were actually hosten on a city of Oakland server this would generally match the process it would go to. 

The trash detection method lives within `find_trashes` inside `server.py`, and is added to the frontend route `'/api/detect'`. The actual display for the results is contained within `processed-static.html` for the display. The only thing of note is to add the API key in within the `.env` file: `ROBOFLOW_API_KEY=`. 

## Haoyang:

* Designed the dynamic process of detecting illegal dumpings with 3 iterations, the code is on [colab](https://colab.research.google.com/drive/1yD4aRgR36VuThiYNIqRP35prEBQF5KQ1#scrollTo=YsMzcQIOBJr9)
  * I am also working on a much more detailed write up about the process: https://github.com/1234567Yang/IllegalDumpingDynamicCameraDetection/
  * Here is a quick skim of what it will look like (this is only the first version):


<p align="center">
  <img width="600" alt="image" src="https://github.com/user-attachments/assets/d46f3f96-307f-44ba-b367-97a3bc58ce1b" />
  <img width="600" alt="image" src="https://github.com/user-attachments/assets/d2dffac0-b6b4-4408-9415-6c4432c8e8d7" />
  <img width="600" alt="image" src="https://github.com/user-attachments/assets/73ed061e-27b1-472d-a871-44c28ea1d105" />
</p>


### Here is the rough overall description of the process (not in detailed):
```
My first version detects human skeletons and tries to see if hands connect to anything. However, I abandoned this method after some testing. First of all, because it's from the camera's angle, the skeletons are not that clear. Secondly, it requires someone to hold the trash continuously for 3 seconds to avoid false positives, which is not very practical.

The second version I swiched to SAM2 and use the finger edge as the detection point, while the trash might move left and right causing the point is not stable. I tried to switch to multipoints labeling for SAM2, but it costs too much time and it's not stable as well. The skeleton detection also not works too well.

In this case, I switched to the current method:

Two foreground extractors named A and B. I use the CNT foreground extractor. A has a fast learning rate, and B has a slow learning rate, which means that A can absorb new objects faster and B is slower.

When a person enters the area, YOLO will assign that person an ID, and after that ID has left the area for a certain number of seconds, we can say that that person has disappeared. However, the YOLO tracking algorithm is not as good as I thought, so I also use CNT A to enhance this tracking process.

Once the person has disappeared, we can look through their path and see if there is any change, as CNT B has a slow learning rate and new objects have not been absorbed yet.

If there is a change, there are two situations:

    A new object
    A disappeared object


To handle this, we first use YOLO to "whitelist" some common objects, like cars and bicycles. After that, I use contour extraction to see if the object has very complex contours, as trash usually has much more complex contours than the background. I also use color comparison to see if that area has a very unusual color, as trash usually comes in all different kinds of colors.

If the confidence is higher than 0.6, the program will trigger the alert.

This process can reach at least 4.5 frames per second, and as fast as 12 frames per second depends on the number of people in the scene.

After that, I noiced the YoloV26l model is not so good at recognizing people from the camera angle, so I tried to look for datasets online. There are two datasets I found: VisDrone and OD-VIRAT Tiny. After I got both datasets, I found that OD-VIRAT Tiny has a lot of similar frames, which can cause overfitting. In this case, I only used Visdrone to train the model. I first trained a large model, and the result is good. However, I tried to let the inference be faster, so I also trained a small model with Visdrone, but the result is not so good and it false positives a lot.

In the end, I chose to use the large model.
```

* Designed an API backend which can be used in the future: See [1234567Yang/IllegalDumpingAPIBackend](https://github.com/1234567Yang/IllegalDumpingAPIBackend)
* Designed the database structure and write the code (MySQL) with 1-3NF: See [database/](database/)


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
