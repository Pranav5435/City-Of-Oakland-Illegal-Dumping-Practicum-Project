import os
import cv2
import base64
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from inference_sdk import InferenceHTTPClient

load_dotenv()

app = Flask(__name__)
CORS(app)

api_key = os.getenv('ROBOFLOW_API_KEY')

client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=api_key
)


def find_trashes(image_path: str, min_confidence: float = 0.5):
    result = client.run_workflow(
        workspace_name="oakland-trash-detection",
        workflow_id="find-trashes-8",
        images={"image": image_path},
        use_cache=True
    )

    img = cv2.imread(image_path)
    predictions = result[0]["predictions"]["predictions"]
    count = 0

    for det in predictions:
        if det["confidence"] < min_confidence:
            continue
        count += 1
        x1 = int(det["x"] - det["width"] / 2)
        y1 = int(det["y"] - det["height"] / 2)
        x2 = int(det["x"] + det["width"] / 2)
        y2 = int(det["y"] + det["height"] / 2)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f'{det["class"]} {det["confidence"]:.2f}'
        cv2.putText(img, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    confident = [d for d in predictions if d["confidence"] >= min_confidence]
    avg_confidence = sum(d["confidence"] for d in confident) / len(confident) if confident else 0

    return img, count, avg_confidence


@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})


@app.route('/api/detect', methods=['POST'])
def detect():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    min_confidence = float(request.form.get('min_confidence', 0.5))

    # Decode uploaded image into OpenCV format
    file_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # Save temporarily so the Roboflow SDK can read it from disk
    temp_path = 'temp_upload.jpg'
    cv2.imwrite(temp_path, img)

    try:
        img_annotated, count, avg_confidence = find_trashes(temp_path, min_confidence)

        # Encode annotated image as base64 to send to the browser
        _, buffer = cv2.imencode('.jpg', img_annotated)
        img_base64 = base64.b64encode(buffer).decode('utf-8')

        return jsonify({
            'annotated_image': img_base64,
            'count': count,
            'average_confidence': round(avg_confidence, 4)
        })

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


if __name__ == '__main__':
    app.run(debug=True, port=8000)