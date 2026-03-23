import os
import cv2
import base64
import numpy as np
import json
import threading
import time
import subprocess
from pathlib import Path
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from dotenv import load_dotenv
from inference_sdk import InferenceHTTPClient

import torch
from ultralytics import YOLO
from boxmot import StrongSort

load_dotenv()

app = Flask(__name__)
CORS(app)

api_key = os.getenv('ROBOFLOW_API_KEY')

client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key=api_key
)

video_jobs = {}

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ========================================
# Static image detection
# ========================================

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


# ========================================
# Video detection pipeline
# ========================================

def run_video_detection(job_id: str, video_path: str, skip_frames: int = 9):
    ffmpeg_proc = None
    try:
        video_jobs[job_id]['status'] = 'running'
        print(f"[job {job_id}] Starting...")

        # ---- Models ----
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model  = YOLO('yolov8n.pt').to(device)
        tracker = StrongSort(
            reid_weights=Path(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'osnet_x0_25_msmt17.pt')),
            device='0' if torch.cuda.is_available() else 'cpu',
            half=True,
            max_cos_dist=0.15,
            max_iou_dist=1.0,
            n_init=4,
            min_conf=0.2,
        )

        # ---- Video info ----
        cap          = cv2.VideoCapture(video_path)
        fps          = cap.get(cv2.CAP_PROP_FPS) or 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        orig_w       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"[job {job_id}] Video: {orig_w}x{orig_h} @ {fps:.1f}fps, {total_frames} frames")

        # Scale to 480 wide, cap height for portrait videos
        TARGET_W = 480
        TARGET_H = int(orig_h * TARGET_W / orig_w)
        TARGET_H = min(TARGET_H, 854)
        TARGET_H = TARGET_H if TARGET_H % 2 == 0 else TARGET_H + 1
        w, h = TARGET_W, TARGET_H
        out_fps = fps / skip_frames
        print(f"[job {job_id}] Processing at {w}x{h}, output {out_fps:.1f}fps")

        # ---- ffmpeg setup (BEFORE the main loop) ----
        # Output is a single frame the same size as the input (w x h)
        # The quad view is scaled down to fit in that same footprint
        output_video_path = os.path.join(OUTPUT_DIR, f'{job_id}.mp4')
        ffmpeg_cmd = [
            'ffmpeg', '-y',
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-s', f'{w}x{h}',   # same size as original
            '-pix_fmt', 'bgr24',
            '-r', str(out_fps),
            '-i', '-',
            '-c:v', 'libx264',
            '-preset', 'ultrafast',
            '-crf', '28',
            '-pix_fmt', 'yuv420p',
            output_video_path
        ]
        print(f"[job {job_id}] Starting ffmpeg -> {output_video_path}")
        try:
            ffmpeg_proc = subprocess.Popen(
                ffmpeg_cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            # Give ffmpeg half a second to start, then check it didn't crash
            time.sleep(0.5)
            if ffmpeg_proc.poll() is not None:
                raise RuntimeError(f'ffmpeg crashed on startup (exit code {ffmpeg_proc.returncode})')
            print(f"[job {job_id}] ffmpeg started ok (pid {ffmpeg_proc.pid})")
        except FileNotFoundError:
            raise RuntimeError('ffmpeg not found on PATH.')

        # ---- Background subtractors ----
        cnt_fast = cv2.bgsegm.createBackgroundSubtractorCNT(
            minPixelStability=5, maxPixelStability=300, useHistory=True, isParallel=True)
        cnt_slow = cv2.bgsegm.createBackgroundSubtractorCNT(
            minPixelStability=80, maxPixelStability=1000, useHistory=True, isParallel=True)

        def _ks(b):
            s = max(3, b)
            return s if s % 2 == 1 else s + 1

        k_open   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_ks(3), _ks(3)))
        k_close  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_ks(9), _ks(9)))
        k_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_ks(7), _ks(7)))

        def clean_fg(fg):
            fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN,  k_open)
            fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, k_close, iterations=2)
            fg = cv2.dilate(fg, k_dilate, iterations=2)
            fg = cv2.medianBlur(fg, 5)
            contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            result = np.zeros_like(fg)
            for c in contours:
                if cv2.contourArea(c) >= 250:
                    cv2.drawContours(result, [c], -1, 255, -1)
            return result

        # ---- Tracking state ----
        people_paths     = {}
        people_tips      = {}
        people_init_box  = {}
        people_last_seen = {}
        people_first_seen= {}
        disappeared      = set()
        historical_frames= {}
        detection_events = []
        frames_processed = 0
        total_to_process = max(1, total_frames // skip_frames)
        frame_idx        = 0

        print(f"[job {job_id}] Entering main loop ({total_to_process} frames to process)...")

        # ---- Main loop ----
        while True:
            for _ in range(skip_frames - 1):
                if not cap.grab():
                    break

            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_LINEAR)

            if frame_idx % (50 * skip_frames) == 0:
                historical_frames[frame_idx] = frame.copy()
            while len(historical_frames) > 100:
                del historical_frames[min(historical_frames)]

            fg_fast = clean_fg(cnt_fast.apply(frame, learningRate=0.1))
            fg_slow = clean_fg(cnt_slow.apply(frame))

            if frame_idx > 250:
                total_px = w * h
                if cv2.countNonZero(fg_fast) / total_px > 0.65:
                    frame_idx += skip_frames
                    continue
                if cv2.countNonZero(fg_slow) / total_px > 0.85:
                    frame_idx += skip_frames
                    continue

            # YOLO + tracker
            results = model(frame, classes=[0], conf=0.4, verbose=False)[0]
            detections_raw = []
            if results.boxes is not None and len(results.boxes) > 0:
                boxes   = results.boxes.xyxy.cpu().numpy()
                scores  = results.boxes.conf.cpu().numpy()
                classes = results.boxes.cls.cpu().numpy().astype(int)
                if len(boxes) > 0:
                    dets   = np.hstack([boxes, scores.reshape(-1, 1), classes.reshape(-1, 1)])
                    tracks = tracker.update(dets, frame)
                    for t in tracks:
                        x1, y1, x2, y2 = int(t[0]), int(t[1]), int(t[2]), int(t[3])
                        tid = int(t[4])
                        detections_raw.append({'id': tid, 'box': np.array([x1, y1, x2, y2])})
                else:
                    tracker.update(np.empty((0, 6)), frame)
            else:
                tracker.update(np.empty((0, 6)), frame)

            # Update walking paths
            for p in detections_raw:
                pid = p['id']
                if pid in disappeared:
                    continue
                people_last_seen[pid] = frame_idx
                if pid not in people_paths:
                    people_init_box[pid]  = p['box']
                    people_first_seen[pid]= frame_idx
                    init_fg = np.zeros((h, w), dtype=np.uint8)
                    x1, y1, x2, y2 = p['box']
                    cv2.rectangle(init_fg, (x1, y1), (x2, y2), 255, -1)
                    people_paths[pid] = init_fg
                    people_tips[pid]  = init_fg.copy()

            # Check for disappeared people
            ids_to_check = [
                pid for pid, last in people_last_seen.items()
                if pid not in disappeared and (frame_idx - last) / fps >= 7
            ]

            for pid in ids_to_check:
                path_mask = people_paths[pid]
                init_box  = people_init_box[pid]
                region    = cv2.subtract(path_mask.copy(), fg_fast)
                leftover  = cv2.bitwise_and(region, fg_slow)
                box_area  = (init_box[2] - init_box[0]) * (init_box[3] - init_box[1])
                min_area  = box_area * 0.15
                contours, _ = cv2.findContours(leftover, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                significant = [c for c in contours if cv2.contourArea(c) >= min_area]

                if significant:
                    timestamp_sec = round(frame_idx / fps, 2)
                    regions = []
                    for c in significant:
                        x, y, bw, bh = cv2.boundingRect(c)
                        regions.append({'x': x, 'y': y, 'w': bw, 'h': bh,
                                        'area': int(cv2.contourArea(c))})
                    _, buf = cv2.imencode('.jpg', frame)
                    frame_b64 = base64.b64encode(buf).decode('utf-8')
                    detection_events.append({
                        'person_id':     int(pid),
                        'frame':         int(frame_idx),
                        'timestamp_sec': timestamp_sec,
                        'regions':       regions,
                        'snapshot':      frame_b64
                    })
                    print(f"[job {job_id}] DETECTION at {timestamp_sec}s, pid={pid}")

                disappeared.add(pid)
                for d in [people_paths, people_tips, people_last_seen,
                          people_init_box, people_first_seen]:
                    d.pop(pid, None)

            # ---- Build quad view scaled to w x h ----
            # Each of the 4 panels is w//2 x h//2
            pw, ph = w // 2, h // 2

            def make_panel(img):
                """Resize any frame to panel size."""
                return cv2.resize(img, (pw, ph), interpolation=cv2.INTER_LINEAR)

            # Panel 1 (top-left): YOLO tracking
            p1 = frame.copy()
            for p in detections_raw:
                x1, y1, x2, y2 = p['box']
                cv2.rectangle(p1, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(p1, f'ID:{p["id"]}', (x1, max(y1 - 6, 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
            cv2.putText(p1, 'YOLO Tracking', (6, 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Panel 2 (top-right): detection results + path overlays
            p2 = frame.copy()
            for pid, path in people_paths.items():
                mask = path > 0
                p2[mask] = (p2[mask] * 0.6 + np.array([255, 255, 0]) * 0.4).astype(np.uint8)
            for ev in detection_events:
                if ev['frame'] == frame_idx:
                    for r in ev['regions']:
                        cv2.rectangle(p2, (r['x'], r['y']),
                                      (r['x'] + r['w'], r['y'] + r['h']), (0, 0, 255), 2)
                        cv2.putText(p2, 'ALERT', (r['x'], max(r['y'] - 6, 10)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            label2 = 'ALERT' if any(ev['frame'] == frame_idx for ev in detection_events) else 'No Detection'
            color2 = (0, 0, 255) if label2 == 'ALERT' else (0, 255, 0)
            cv2.putText(p2, label2, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color2, 1)

            # Panel 3 (bottom-left): CNT Fast (moving objects)
            p3 = frame.copy()
            fast_mask = fg_fast > 0
            p3[fast_mask] = (p3[fast_mask] * 0.5 + np.array([0, 0, 255]) * 0.5).astype(np.uint8)
            contours_fast, _ = cv2.findContours(fg_fast, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(p3, contours_fast, -1, (255, 255, 255), 1)
            cv2.putText(p3, 'CNT Fast (Motion)', (6, 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Panel 4 (bottom-right): CNT Slow (background changes)
            p4 = frame.copy()
            slow_mask = fg_slow > 0
            p4[slow_mask] = (p4[slow_mask] * 0.5 + np.array([0, 100, 255]) * 0.5).astype(np.uint8)
            contours_slow, _ = cv2.findContours(fg_slow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(p4, contours_slow, -1, (255, 255, 255), 1)
            cv2.putText(p4, 'CNT Slow (Changes)', (6, 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Timestamp on all panels
            ts = f'{frame_idx / fps:.1f}s'
            for panel in [p1, p2, p3, p4]:
                cv2.putText(panel, ts, (6, panel.shape[0] - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

            # Assemble quad grid scaled to w x h
            top_row    = cv2.hconcat([make_panel(p1), make_panel(p2)])
            bottom_row = cv2.hconcat([make_panel(p3), make_panel(p4)])
            quad       = cv2.vconcat([top_row, bottom_row])

            # Write to ffmpeg
            ffmpeg_proc.stdin.write(quad.tobytes())

            frames_processed += 1
            frame_idx        += skip_frames
            video_jobs[job_id]['progress'] = round(frames_processed / total_to_process * 100, 1)

        cap.release()
        print(f"[job {job_id}] Main loop done, closing ffmpeg...")

        ffmpeg_proc.stdin.close()
        ffmpeg_proc.wait()
        print(f"[job {job_id}] ffmpeg done, output: {output_video_path}")

        video_jobs[job_id]['status']   = 'complete'
        video_jobs[job_id]['progress'] = 100
        video_jobs[job_id]['result']   = {
            'total_frames_processed': frames_processed,
            'total_detections':       len(detection_events),
            'events':                 detection_events,
            'output_video_job_id':    job_id
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        video_jobs[job_id]['status'] = 'error'
        video_jobs[job_id]['error']  = str(e)

    finally:
        if ffmpeg_proc is not None:
            try:
                ffmpeg_proc.stdin.close()
                ffmpeg_proc.wait()
            except Exception:
                pass
        if os.path.exists(video_path):
            try:
                os.remove(video_path)
            except Exception:
                pass


# ========================================
# Routes
# ========================================

@app.route('/api/video-output/<job_id>', methods=['GET'])
def video_output(job_id):
    path = os.path.join(OUTPUT_DIR, f'{job_id}.mp4')
    if not os.path.exists(path):
        return jsonify({'error': 'Output video not found'}), 404
    return send_file(path, mimetype='video/mp4')


@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})


@app.route('/api/detect', methods=['POST'])
def detect():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file           = request.files['image']
    min_confidence = float(request.form.get('min_confidence', 0.5))

    file_bytes = np.frombuffer(file.read(), np.uint8)
    img        = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    temp_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'temp_upload.jpg')
    cv2.imwrite(temp_path, img)

    try:
        img_annotated, count, avg_confidence = find_trashes(temp_path, min_confidence)
        _, buffer  = cv2.imencode('.jpg', img_annotated)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        return jsonify({
            'annotated_image':    img_base64,
            'count':              count,
            'average_confidence': round(avg_confidence, 4)
        })
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


@app.route('/api/detect-video', methods=['POST'])
def detect_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No video uploaded'}), 400

    file        = request.files['video']
    skip_frames = int(request.form.get('skip_frames', 9))

    temp_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             f'temp_video_{int(time.time())}.mp4')
    file.save(temp_path)

    job_id = f'job_{int(time.time() * 1000)}'
    video_jobs[job_id] = {'status': 'queued', 'progress': 0, 'result': None, 'error': None}

    t = threading.Thread(
        target=run_video_detection,
        args=(job_id, temp_path, skip_frames),
        daemon=True
    )
    t.start()

    return jsonify({'job_id': job_id, 'message': 'Video processing started'})


@app.route('/api/detect-video/<job_id>', methods=['GET'])
def detect_video_status(job_id):
    if job_id not in video_jobs:
        return jsonify({'error': 'Job not found'}), 404
    return jsonify(video_jobs[job_id])


if __name__ == '__main__':
    app.run(debug=True, port=8000)