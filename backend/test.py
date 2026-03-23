import cv2
import numpy as np
import time
import torch
import subprocess
from ultralytics import YOLO

from boxmot import StrongSort
from pathlib import Path

from collections import deque
import matplotlib.pyplot as plt

from skimage.feature import local_binary_pattern

import threading


# ========================================
# Configuration Constants
# ========================================

DEBUG_DETAIL_OUTPUT = True

VIDEO_PATH = 'demo_vid_4.mp4'
OUTPUT_PATH = 'result_simple.mp4'
OUTPUT_H264_PATH = 'result_simple_h264.mp4'

# Target width & scale factor (probe original resolution first)
TARGET_W = 960
_BASE_W = 960  # All thresholds were tuned at this resolution

_probe = cv2.VideoCapture(VIDEO_PATH)
_orig_w = int(_probe.get(cv2.CAP_PROP_FRAME_WIDTH))
_orig_h = int(_probe.get(cv2.CAP_PROP_FRAME_HEIGHT))
_probe.release()

TARGET_H = int(_orig_h * TARGET_W / _orig_w)
TARGET_H = TARGET_H if TARGET_H % 2 == 0 else TARGET_H + 1

# Threshold scale factor: only depends on TARGET_W, not the original video
_S = TARGET_W / _BASE_W          # Length scale: 960→1.0, 1280→1.333
_S2 = _S ** 2                     # Area scale


SKIP_FRAMES = 12        # Process 1 out of every 12 frames
YOLO_MODEL = 'visdrone_yolo26l_best.pt'
YOLO_CONF = 0.7
YOLO_VEHICLE_CONF = 0.4

YOLO_TRACK_CLASSES = [0]                        # Classes to track (people)
YOLO_VEHICLE_CLASSES = [2, 3, 4, 7, 8]         # Vehicle whitelist
YOLO_ALL_CLASSES = list(set(YOLO_TRACK_CLASSES + YOLO_VEHICLE_CLASSES))
UPDATE_WALKING_VEHICLE_BOX_EXPAND = 0.3

# YOLO absolute ignore: if area/width is too small, ignore regardless of confidence
_YOLO_IGNORE_AREA_PERC = (12 * 16) / (1000 * 540)
_YOLO_IGNORE_WIDTH_PERC = 10 / 1000
YOLO_ABSOLUTE_IGNORE_AREA = _YOLO_IGNORE_AREA_PERC * (TARGET_W * TARGET_H)
YOLO_ABSOLUTE_IGNORE_WIDTH = _YOLO_IGNORE_WIDTH_PERC * TARGET_W

HISTORICAL_FRAME_SAVE_PER_FRAME = 50   # Save one frame to history every 50 frames
HISTORICAL_FRAME_MAX_SAVE = 100

# YOLO and person disappearance
YOLO_VERIFY_AS_HUMAN_DISAPPEARED_AFTER_SECOND = 7

# CNT settings
CNT_FAST_IGNORE_FRAME_IF_SUDDEN_CHANGE_EXCEEDS = 0.65  # Ignore frame if >65% of pixels suddenly change (e.g. bird)
CNT_SLOW_IGNORE_FRAME_IF_SUDDEN_CHANGE_EXCEEDS = 0.85
CNT_NO_IGNORE_IN_THE_BEGINNING_FRAME = 250  # Don't ignore the first 250 frames — give it time to warm up

BG_MIN_PIXEL_STABILITY_CNT_FAST = 5
BG_MAX_PIXEL_STABILITY_CNT_FAST = 300
CNT_FAST_BOOST_LEARNING = 0.1

# A pixel must remain stable for N frames before CNT treats it as background.
# At 30fps with skip=5, that's 50×5=250 frames ≈ 8.3 seconds of stillness before absorption.
# This means a person standing still for under 8 seconds won't be swallowed into the background.
BG_MIN_PIXEL_STABILITY_CNT_SLOW = 80
# Once a pixel is confirmed as background, its stability count caps at this value.
# Higher = more "stubborn" background, harder to overwrite with new content.
BG_MAX_PIXEL_STABILITY_CNT_SLOW = 1000

cntFast = cv2.bgsegm.createBackgroundSubtractorCNT(
    minPixelStability=BG_MIN_PIXEL_STABILITY_CNT_FAST,
    maxPixelStability=BG_MAX_PIXEL_STABILITY_CNT_FAST,
    useHistory=True,
    isParallel=True
)

cntSlow = cv2.bgsegm.createBackgroundSubtractorCNT(
    minPixelStability=BG_MIN_PIXEL_STABILITY_CNT_SLOW,
    maxPixelStability=BG_MAX_PIXEL_STABILITY_CNT_SLOW,
    useHistory=True,
    isParallel=True
)

# Kernel helpers: ensure odd kernel sizes
def _ks(base):
    s = max(3, int(base * _S))
    return s if s % 2 == 1 else s + 1

kernel_open   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_ks(3), _ks(3)))
kernel_close  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_ks(9), _ks(9)))
kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_ks(7), _ks(7)))

_RES_SCALE = TARGET_W / _BASE_W
CNT_FG_MIN_AREA = int(250 * _S2)  # Minimum connected component area

# Dictionary remap thresholds
DICT_REMAP_TIP_IN_BOX_FILL = 0.7           # If 70% of a box is covered by a tracker tip, remap
DICT_REMAP_TIP_IN_BOX_RATIO = 0.9          # If the tracker tip is 90% inside a box, remap
DICT_REMAP_TIP_IN_BOX_FILL_MIN_RATIO = 0.3 # Minimum ratio when fill condition passes (prevents small-box false positives)

# Walking path update settings
UPDATE_WALKING_PATH_EXPANSION = 2
UPDATE_WALKING_FG_FAST_WORKING_AREA_EXPAND = 1.8         # Fast FG only works in a small region
UPDATE_WALKING_FG_FAST_WORKING_AREA_EXPAND_MAXIMUM = int(350 * _S)
UPDATE_WALKING_PATH_INTERSECTION_MIN_SIZE_PERCENTAGE = 0.02
UPDATE_WALKING_PATH_TARGET_MAX_DIFF_PERCENTAGE = 6.0     # Discard if new size differs too much
UPDATE_WALKING_PATH_INIT_BOX_MAX_DECREASED_PERCENT = 0.2 # Use init box if path shrinks below 20% of it
UPDATE_WALKING_WHITELIST_EXPANSION = 1.55
UPDATE_WALKING_PATH_BLOB_MIN_RATIO_TO_LARGEST = 0.3      # Discard blobs smaller than 30% of the largest
UPDATE_WALKING_PATH_BLOB_MIN_AREA = int(225 * _S2)       # Absolute minimum blob area

UPDATE_WALKING_PATH_TRACKER_RM_EDGE_MARGIN = 2           # Pixels from edge that counts as "reached edge"
UPDATE_WALKING_PATH_EDGE_CLEAR_MIN_MISSING_FRAMES = 10   # Only check edge-clearing after 10 missing frames
UPDATE_WALKING_PATH_EDGE_CLEAR_MIN_PATH_RATIO = 2.5      # Path area must be 2.5x the initial box

# Final detection settings
DETECT_PEOPLE_PATH_EXPANSION = 1.66
DETECT_CNT_FAST_FG_CUT_EXPANSION = 1.5     # Cut out currently-moving objects
DETECT_RESULT_POSITIVE_REL_SIZE = 0.15     # Must be >15% of original box area to count as positive
CHECK_AREA_MERGE_CLOSE_KERNEL_SIZE = _ks(21)

# Left-area YOLO whitelist settings
CHECK_AREA_YOLO_WHITELIST_BOX_EXPAND = 0.75
CHECK_AREA_YOLO_WHITELIST_CONF = 0.3
CHECK_AREA_YOLO_DETECT_CLASS = [0, 1, 2, 3, 4, 7, 8]  # person, bicycle, car, van, truck, bus, motor
CHECK_AREA_AFTER_WHITELIST_SMALLER_THRESHOLD = 0.1


# ========================================
# Output: JSON summary of detections
# ========================================
import json
from datetime import datetime

detection_log = []  # List of detection events to save at the end


def save_detection_log():
    """Save all detection events to a JSON file."""
    output = {
        "video": VIDEO_PATH,
        "processed_at": datetime.now().isoformat(),
        "total_detections": len(detection_log),
        "events": detection_log
    }
    log_path = OUTPUT_H264_PATH.replace('.mp4', '_detections.json')
    with open(log_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"Detection log saved to: {log_path}")


def log_detection_event(pid, frame_idx, fps, merged_masks):
    """Record a single dumping detection event."""
    timestamp_sec = round(frame_idx / fps, 2)
    bboxes = []
    for mask in merged_masks:
        coords = cv2.findNonZero(mask)
        if coords is not None:
            x, y, w, h = cv2.boundingRect(coords)
            bboxes.append({"x": x, "y": y, "w": w, "h": h})
    detection_log.append({
        "person_id": int(pid),
        "frame": int(frame_idx),
        "timestamp_sec": timestamp_sec,
        "bounding_boxes": bboxes
    })
    print(f"  [detection logged] pid={pid}, t={timestamp_sec}s, {len(bboxes)} region(s)")


# ========================================
# Core Functions
# ========================================

def process_single_frame(frame, current_frame_number, model, tracker):
    # Infer with lower conf, then filter by class separately
    min_conf = min(YOLO_CONF, YOLO_VEHICLE_CONF)
    results = model(frame, classes=YOLO_ALL_CLASSES, conf=min_conf, verbose=False, half=True)[0]

    persons = []
    img_h, img_w = frame.shape[:2]
    vehicle_mask = np.zeros((img_h, img_w), dtype=np.uint8)

    if results.boxes is not None and len(results.boxes) > 0:
        boxes   = results.boxes.xyxy.cpu().numpy()
        scores  = results.boxes.conf.cpu().numpy()
        classes = results.boxes.cls.cpu().numpy().astype(int)

        # Separate vehicles, filter by YOLO_VEHICLE_CONF, paint onto vehicle_mask
        vehicle_idx = np.isin(classes, YOLO_VEHICLE_CLASSES) & (scores >= YOLO_VEHICLE_CONF)
        for box in boxes[vehicle_idx]:
            x1, y1, x2, y2 = expand_box(
                int(box[0]), int(box[1]), int(box[2]), int(box[3]),
                UPDATE_WALKING_VEHICLE_BOX_EXPAND, img_h, img_w
            )
            cv2.rectangle(vehicle_mask, (x1, y1), (x2, y2), 255, -1)

        # Send only people to the tracker, filtered by YOLO_CONF
        person_idx = np.isin(classes, YOLO_TRACK_CLASSES) & (scores >= YOLO_CONF)
        p_boxes   = boxes[person_idx]
        p_scores  = scores[person_idx]
        p_classes = classes[person_idx]

        # Filter out detections that are too small
        widths  = p_boxes[:, 2] - p_boxes[:, 0]
        heights = p_boxes[:, 3] - p_boxes[:, 1]
        areas   = widths * heights
        keep = (areas >= YOLO_ABSOLUTE_IGNORE_AREA) & (widths >= YOLO_ABSOLUTE_IGNORE_WIDTH) & (heights >= YOLO_ABSOLUTE_IGNORE_WIDTH)
        p_boxes   = p_boxes[keep]
        p_scores  = p_scores[keep]
        p_classes = p_classes[keep]

        if len(p_boxes) > 0:
            # Stack into [x1,y1,x2,y2,conf,cls] Nx6 array and pass to StrongSort
            dets   = np.hstack([p_boxes, p_scores.reshape(-1, 1), p_classes.reshape(-1, 1)])
            tracks = tracker.update(dets, frame)
            for t in tracks:
                x1, y1, x2, y2 = int(t[0]), int(t[1]), int(t[2]), int(t[3])
                tid   = int(t[4])
                score = float(t[5])
                persons.append({"id": tid, "box": np.array([x1, y1, x2, y2]), "score": score})
        else:
            tracker.update(np.empty((0, 6)), frame)
    else:
        tracker.update(np.empty((0, 6)), frame)

    return results, {"frame_id": current_frame_number, "persons": persons}, vehicle_mask


def expand_fg(fg, ratio=1.0):
    """
    Dilate a foreground mask by area ratio.
    ratio=2.0 means the white pixel area roughly doubles.
    Works on any irregular shape.
    """
    if ratio <= 1.0:
        return fg.copy()

    area = cv2.countNonZero(fg)
    if area == 0:
        return fg.copy()

    r  = np.sqrt(area / np.pi)
    px = int(r * (np.sqrt(ratio) - 1))
    if px <= 0:
        return fg.copy()

    # Only work within the bounding box + margin to avoid full-frame distance transform
    coords = cv2.findNonZero(fg)
    if coords is None:
        return fg.copy()
    bx, by, bw, bh = cv2.boundingRect(coords)
    pad = px + 2
    rx1 = max(0, bx - pad)
    ry1 = max(0, by - pad)
    rx2 = min(fg.shape[1], bx + bw + pad)
    ry2 = min(fg.shape[0], by + bh + pad)

    roi = fg[ry1:ry2, rx1:rx2]
    inv  = cv2.bitwise_not(roi)
    dist = cv2.distanceTransform(inv, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)

    _, roi_result = cv2.threshold(dist, px, 255, cv2.THRESH_BINARY_INV)
    roi_result = roi_result.astype(np.uint8)

    result = fg.copy()
    result[ry1:ry2, rx1:rx2] = cv2.bitwise_or(result[ry1:ry2, rx1:rx2], roi_result)
    return result


def update_walking_path(walking_path_fg, last_updated_fg, fg_fast, fg_fast_white_list, init_box=None):
    """
    Extend a person's walking path by intersecting their last known tip
    with the current fast foreground, restricted to a local ROI for efficiency.
    """
    coords = cv2.findNonZero(last_updated_fg)
    if coords is None:
        return walking_path_fg, last_updated_fg

    x, y, bw, bh = cv2.boundingRect(coords)

    margin = int(max(bw, bh) * UPDATE_WALKING_FG_FAST_WORKING_AREA_EXPAND)
    margin = min(margin, UPDATE_WALKING_FG_FAST_WORKING_AREA_EXPAND_MAXIMUM)

    rx1 = max(0, x - margin)
    ry1 = max(0, y - margin)
    rx2 = min(fg_fast.shape[1], x + bw + margin)
    ry2 = min(fg_fast.shape[0], y + bh + margin)

    tip_roi   = last_updated_fg[ry1:ry2, rx1:rx2]
    fast_roi  = fg_fast[ry1:ry2, rx1:rx2]

    # Exclude whitelist regions (e.g. other people's tips)
    whitelist_roi = fg_fast_white_list[ry1:ry2, rx1:rx2]
    whitelist_roi = expand_fg(whitelist_roi, UPDATE_WALKING_WHITELIST_EXPANSION)
    fast_roi = cv2.subtract(fast_roi, whitelist_roi)

    tip_expanded      = expand_fg(tip_roi,  ratio=UPDATE_WALKING_PATH_EXPANSION)
    fast_roi_expanded = expand_fg(fast_roi, ratio=UPDATE_WALKING_PATH_EXPANSION)

    intersection = cv2.bitwise_and(tip_expanded, fast_roi_expanded)
    inter_area   = cv2.countNonZero(intersection)
    tip_area     = cv2.countNonZero(tip_expanded)

    if DEBUG_DETAIL_OUTPUT:
        tip_pixels  = cv2.countNonZero(last_updated_fg)
        fast_pixels = cv2.countNonZero(fg_fast)
        path_pixels = cv2.countNonZero(walking_path_fg)
        if tip_pixels > 20000 or fast_pixels > 20000:
            print(f"[update] path={path_pixels}, tip={tip_pixels}, fg_fast={fast_pixels}")

    # Find all connected components in fast_roi that intersect with the tip
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(fast_roi, connectivity=8)
    overall_intersection = cv2.bitwise_and(fast_roi_expanded, tip_expanded)
    matched_blobs = np.zeros_like(fast_roi)
    for i in range(1, num_labels):
        x_s, y_s, w_s, h_s, area_s = stats[i]
        margin_s = 20
        x1_s = max(0, x_s - margin_s)
        y1_s = max(0, y_s - margin_s)
        x2_s = min(fast_roi.shape[1], x_s + w_s + margin_s)
        y2_s = min(fast_roi.shape[0], y_s + h_s + margin_s)
        overlap = cv2.countNonZero(overall_intersection[y1_s:y2_s, x1_s:x2_s])
        if overlap > 0:
            matched_blobs[labels == i] = 255

    if cv2.countNonZero(matched_blobs) == 0:
        return walking_path_fg, last_updated_fg

    # Filter out blobs that are too small relative to the largest matched blob
    num_m, labels_m, stats_m, _ = cv2.connectedComponentsWithStats(matched_blobs, connectivity=8)
    max_blob_area = max(stats_m[i, cv2.CC_STAT_AREA] for i in range(1, num_m)) if num_m > 1 else 0
    filtered_blobs = np.zeros_like(matched_blobs)
    for i in range(1, num_m):
        area_i = stats_m[i, cv2.CC_STAT_AREA]
        if area_i >= max_blob_area * UPDATE_WALKING_PATH_BLOB_MIN_RATIO_TO_LARGEST and area_i >= UPDATE_WALKING_PATH_BLOB_MIN_AREA:
            filtered_blobs[labels_m == i] = 255

    if cv2.countNonZero(filtered_blobs) == 0:
        return walking_path_fg, last_updated_fg

    best_blob = filtered_blobs
    if best_blob is None:
        return walking_path_fg, last_updated_fg

    # Size sanity check
    last_area = cv2.countNonZero(last_updated_fg)
    new_area  = cv2.countNonZero(best_blob)
    if init_box is not None:
        init_area      = (init_box[2] - init_box[0]) * (init_box[3] - init_box[1])
        reference_area = max(last_area, int(init_area * UPDATE_WALKING_PATH_INIT_BOX_MAX_DECREASED_PERCENT))
    else:
        reference_area = last_area
    if reference_area > 0 and (
        new_area > reference_area * (1 + UPDATE_WALKING_PATH_TARGET_MAX_DIFF_PERCENTAGE) or
        new_area < reference_area / (1 + UPDATE_WALKING_PATH_TARGET_MAX_DIFF_PERCENTAGE)
    ):
        return walking_path_fg, last_updated_fg

    if new_area == 0:
        return walking_path_fg, last_updated_fg

    # Paste the local result back onto the full-frame mask
    new_segment = np.zeros_like(fg_fast)
    new_segment[ry1:ry2, rx1:rx2] = best_blob

    new_walking_path_fg = cv2.bitwise_or(walking_path_fg, new_segment)
    return new_walking_path_fg, new_segment


def process_raw_cnt_fg(fg):
    """Clean up raw CNT foreground mask with morphological operations."""
    # 1. Opening: erode then dilate to remove small noise
    clean = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel_open)
    # 2. Closing: fill internal holes and merge nearby fragments
    clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel_close, iterations=2)
    # 3. Light dilation: merge fragments from the same object
    clean = cv2.dilate(clean, kernel_dilate, iterations=2)
    # 4. Median blur: remove remaining salt-and-pepper noise
    clean = cv2.medianBlur(clean, 5)
    # 5. Connected component filter: discard regions that are too small
    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result = np.zeros_like(fg)
    for cnt in contours:
        if cv2.contourArea(cnt) >= CNT_FG_MIN_AREA:
            cv2.drawContours(result, [cnt], -1, 255, -1)
    return result


def detect_anything_left(people_path_fg_mask, cnt_fast_fg_mask, cnt_slow_fg_mask, original_people_box):
    """
    Check if anything was left behind along a person's path.

    people_path_fg_mask:  mask of the path this person walked
    cnt_fast_fg_mask:     current frame's fast CNT foreground (things currently moving)
    cnt_slow_fg_mask:     current frame's slow CNT foreground (things not in original background)
    original_people_box:  bounding box from first detection, used for relative size comparison

    Returns: (is_positive, result_masks, visual_info)
    """
    # 1. Expand the walking path
    path_expanded = expand_fg(people_path_fg_mask, ratio=DETECT_PEOPLE_PATH_EXPANSION)
    path_expanded = process_raw_cnt_fg(path_expanded)

    # 2. Expand currently-moving objects (to subtract them out)
    fast_expanded = expand_fg(cnt_fast_fg_mask, ratio=DETECT_CNT_FAST_FG_CUT_EXPANSION)

    # 3. Path area minus currently-moving area
    region = cv2.subtract(path_expanded, fast_expanded)

    # 4. Intersect with slow CNT (things that weren't in the original background)
    result = cv2.bitwise_and(region, cnt_slow_fg_mask)

    # 5. Size threshold check
    box_area = (original_people_box[2] - original_people_box[0]) * (original_people_box[3] - original_people_box[1])
    min_area = box_area * DETECT_RESULT_POSITIVE_REL_SIZE

    contours, _ = cv2.findContours(result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detections     = []
    areas_too_small = []
    for cnt in contours:
        mask = np.zeros_like(result)
        cv2.drawContours(mask, [cnt], -1, 255, -1)
        if cv2.contourArea(cnt) >= min_area:
            detections.append(mask)
        else:
            areas_too_small.append(mask)

    info_for_visual = {
        "path_expanded":    path_expanded,
        "cnt_fast_expanded": fast_expanded,
        "areas_too_small":  areas_too_small
    }

    return len(detections) > 0, detections, info_for_visual


def expand_box(x1, y1, x2, y2, ratio, img_h, img_w):
    """Expand a bounding box by ratio, clamped to image bounds."""
    bw = x2 - x1
    bh = y2 - y1
    dx = int(bw * ratio / 2)
    dy = int(bh * ratio / 2)
    return max(0, x1 - dx), max(0, y1 - dy), min(img_w, x2 + dx), min(img_h, y2 + dy)


def check_if_left_area_is_trash(frame, history_frame, fg_left, current_frame_yolo_combined_mask, yolo_model):
    """
    Determine whether a suspicious leftover region is actual dumped trash.

    Returns: (is_trash, current_frame_yolo_mask, history_frame_yolo_mask)
    """
    original_area = cv2.countNonZero(fg_left)
    if original_area == 0:
        print("Invalid input: fg_left is empty!")
        return False, None, None

    img_h, img_w = frame.shape[:2]

    if current_frame_yolo_combined_mask is None:
        # Run YOLO on current frame to build a whitelist of recognizable objects
        results_current = yolo_model(frame, conf=CHECK_AREA_YOLO_WHITELIST_CONF,
                                     classes=CHECK_AREA_YOLO_DETECT_CLASS, verbose=False, half=True)[0]
        yolo_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        if results_current.boxes is not None and len(results_current.boxes) > 0:
            boxes = results_current.boxes.xyxy.cpu().numpy().astype(int)
            for box in boxes:
                x1, y1, x2, y2 = expand_box(*box, CHECK_AREA_YOLO_WHITELIST_BOX_EXPAND, img_h, img_w)
                cv2.rectangle(yolo_mask, (x1, y1), (x2, y2), 255, -1)
    else:
        yolo_mask = current_frame_yolo_combined_mask

    # Run YOLO on the historical frame too
    results_history = yolo_model(history_frame, conf=CHECK_AREA_YOLO_WHITELIST_CONF,
                                 classes=CHECK_AREA_YOLO_DETECT_CLASS, verbose=False, half=True)[0]
    yolo_mask_history = np.zeros(history_frame.shape[:2], dtype=np.uint8)
    if results_history.boxes is not None and len(results_history.boxes) > 0:
        boxes = results_history.boxes.xyxy.cpu().numpy().astype(int)
        for box in boxes:
            x1, y1, x2, y2 = expand_box(*box, CHECK_AREA_YOLO_WHITELIST_BOX_EXPAND, img_h, img_w)
            cv2.rectangle(yolo_mask_history, (x1, y1), (x2, y2), 255, -1)

    # Combine both frame whitelists and subtract from suspicious area
    yolo_mask_combined = cv2.bitwise_or(yolo_mask, yolo_mask_history)
    fg_remaining = cv2.subtract(fg_left, yolo_mask_combined)
    remaining_area = cv2.countNonZero(fg_remaining)

    if DEBUG_DETAIL_OUTPUT:
        fig, axes = plt.subplots(1, 5, figsize=(30, 6))
        vis0 = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2RGB)
        vis0[fg_left > 0] = [255, 0, 0]
        axes[0].imshow(vis0); axes[0].set_title(f"fg_left (area={original_area})")

        vis1 = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2RGB)
        vis1[yolo_mask > 0] = [0, 255, 0]
        axes[1].imshow(vis1); axes[1].set_title(f"yolo_mask_current ({cv2.countNonZero(yolo_mask)}px)")

        vis2 = cv2.cvtColor(history_frame.copy(), cv2.COLOR_BGR2RGB)
        vis2[yolo_mask_history > 0] = [0, 0, 255]
        axes[2].imshow(vis2); axes[2].set_title(f"yolo_mask_history ({cv2.countNonZero(yolo_mask_history)}px)")

        vis3 = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2RGB)
        vis3[yolo_mask_combined > 0] = [255, 255, 0]
        vis3[fg_left > 0] = [255, 0, 0]
        overlap = cv2.bitwise_and(fg_left, yolo_mask_combined)
        vis3[overlap > 0] = [255, 0, 255]
        axes[3].imshow(vis3); axes[3].set_title(f"yellow=whitelist, red=fg_left, purple=overlap ({cv2.countNonZero(overlap)}px)")

        vis4 = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2RGB)
        vis4[fg_remaining > 0] = [255, 0, 0]
        axes[4].imshow(vis4); axes[4].set_title(f"fg_remaining ({remaining_area}px, {remaining_area/max(original_area,1)*100:.1f}%)")

        for ax in axes: ax.axis('off')
        plt.tight_layout(); plt.show()
        print(f"  [whitelist debug] fg_left={original_area}, overlap={cv2.countNonZero(overlap)}, remaining={remaining_area}")

    if remaining_area == 0:
        return False, yolo_mask, yolo_mask_history

    # If nearly all of the area was explained by YOLO, it's not trash
    if original_area > 0 and remaining_area < original_area * CHECK_AREA_AFTER_WHITELIST_SMALLER_THRESHOLD:
        return False, yolo_mask, yolo_mask_history

    # ---- Texture/color similarity analysis ----
    # NOTE: This section was AI-generated — review carefully if bugs appear

    CHECK_AREA_MIN_COMPONENT_AREA = int(250 * _S2)
    CHECK_AREA_SIMILARITY_NEW_OBJECT_THRESHOLD = 0.35

    h, w = frame.shape[:2]
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(fg_remaining, connectivity=8)

    new_object_votes        = []
    new_object_vote_weights = []

    for i in range(1, num_labels):
        comp_area = stats[i, cv2.CC_STAT_AREA]
        if comp_area < CHECK_AREA_MIN_COMPONENT_AREA:
            continue

        comp_mask = ((labels == i).astype(np.uint8)) * 255

        # Adaptive dilation radius based on region size
        dilate_radius = int(np.clip(np.sqrt(comp_area) * 0.3, 15, 80))
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (dilate_radius * 2 + 1, dilate_radius * 2 + 1)
        )
        dilated = cv2.dilate(comp_mask, kernel)

        # Surrounding ring = dilated minus all suspicious regions
        surround_mask = cv2.subtract(dilated, fg_remaining)
        surround_area = cv2.countNonZero(surround_mask)

        # Fallback to global background if surround is too small (large dumping event)
        min_required_surround = max(comp_area * 0.1, CHECK_AREA_MIN_COMPONENT_AREA)
        if surround_area < min_required_surround:
            surround_mask = cv2.bitwise_not(fg_remaining)
            surround_area = cv2.countNonZero(surround_mask)

        if surround_area < CHECK_AREA_MIN_COMPONENT_AREA:
            # Can't sample enough background — conservatively treat as new object
            new_object_votes.append(True)
            new_object_vote_weights.append(comp_area)
            continue

        # Compare suspicious region vs surrounding ring
        # sim_score: 0 = identical, 1 = completely different
        sim_score, ct_score, e_score, _, ct_details, e_details, duration = \
            area_comprehensive_similarity(frame, comp_mask, frame, surround_mask)

        is_new = sim_score > CHECK_AREA_SIMILARITY_NEW_OBJECT_THRESHOLD
        new_object_votes.append(is_new)
        new_object_vote_weights.append(comp_area)

        print(f"  [component {i}] area={comp_area}, surround={surround_area}, "
              f"diff={sim_score:.3f}(color {ct_score:.3f}/edge {e_score:.3f}), "
              f"verdict={'new object' if is_new else 'object removed'}, t={duration:.1f}ms")

    if len(new_object_votes) == 0:
        return False, yolo_mask, yolo_mask_history

    # Area-weighted vote: large regions carry more weight than small fragments
    total_weight = sum(new_object_vote_weights)
    new_object_weighted = sum(
        w for v, w in zip(new_object_votes, new_object_vote_weights) if v
    ) / total_weight

    print(f"  [combined verdict] new_object_weighted={new_object_weighted:.2f}")

    # Compare against historical frame to check if texture complexity increased
    CHECK_AREA_HISTORY_COMPLEXITY_BOOST = 0.15

    hist_sim_score, hist_ct_score, hist_e_score, _, hist_ct_details, hist_e_details, hist_duration = \
        area_comprehensive_similarity(frame, fg_remaining, history_frame, fg_remaining)

    current_intensity    = hist_ct_details.get("jump_intensity_a", 0)
    history_intensity    = hist_ct_details.get("jump_intensity_b", 0)
    current_chaos        = hist_ct_details.get("jump_chaos_a", 0)
    history_chaos        = hist_ct_details.get("jump_chaos_b", 0)
    current_richness     = hist_ct_details.get("color_richness_a", 0)
    history_richness     = hist_ct_details.get("color_richness_b", 0)
    current_edge_density = hist_e_details.get("density_a", 0)
    history_edge_density = hist_e_details.get("density_b", 0)

    complexity_increase_count = 0
    if current_intensity    > history_intensity:    complexity_increase_count += 1
    if current_chaos        > history_chaos:        complexity_increase_count += 1
    if current_richness     > history_richness:     complexity_increase_count += 1
    if current_edge_density > history_edge_density: complexity_increase_count += 1

    if complexity_increase_count >= 3:
        boost = CHECK_AREA_HISTORY_COMPLEXITY_BOOST * (complexity_increase_count / 4)
        new_object_weighted = min(1.0, new_object_weighted + boost)
        print(f"  [history compare] complexity increased ({complexity_increase_count}/4 dims), "
              f"+{boost:.3f} boost, adjusted={new_object_weighted:.2f}, t={hist_duration:.1f}ms")
    else:
        print(f"  [history compare] complexity not significantly increased ({complexity_increase_count}/4 dims), "
              f"no boost, t={hist_duration:.1f}ms")

    is_trash = new_object_weighted >= 0.5
    print(f"  [final verdict] weighted={new_object_weighted:.2f}, "
          f"conclusion={'TRASH / new object' if is_trash else 'object removed / false positive'}")

    return is_trash, yolo_mask, yolo_mask_history
    # ---- END AI-generated section ----


# ========================================
# Visualization Helpers
# ========================================

def paint_cnt_overlay(frame, fg_mask, color, label):
    overlay = frame.copy()
    overlay[fg_mask > 0] = color
    blended = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(blended, contours, -1, (255, 255, 255), 1)
    cv2.putText(blended, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return blended


def build_quad_view(frame, yolo_panel, fg_fast, fg_slow,
                    detections=None, visuals=None,
                    active_moving_heads=None, detecting_left_area=None):
    h, w = frame.shape[:2]

    top_left = yolo_panel.copy()
    cv2.putText(top_left, "YOLO Tracking", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    top_right = frame.copy()

    if visuals:
        for vi in visuals:
            contours, _ = cv2.findContours(vi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(top_right, contours, -1, (255, 255, 255), 1)

    if active_moving_heads:
        for head in active_moving_heads:
            top_right[head > 0] = (0, 255, 255)

    if detecting_left_area:
        for area in detecting_left_area:
            contours, _ = cv2.findContours(area["path_expanded"], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(top_right, contours, -1, (255, 255, 0), 1)

    if detections:
        for mask in detections:
            top_right[mask > 0] = (0, 0, 255)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(top_right, contours, -1, (0, 0, 255), 2)
        cv2.putText(top_right, f"ALERT: {len(detections)} object(s)", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    else:
        cv2.putText(top_right, "No detection", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    bottom_left  = paint_cnt_overlay(frame, fg_fast, (0, 0, 255),    "CNT Fast")
    bottom_right = paint_cnt_overlay(frame, fg_slow, (255, 100, 0),  "CNT Slow")

    top_row    = cv2.hconcat([top_left, top_right])
    bottom_row = cv2.hconcat([bottom_left, bottom_right])
    return cv2.vconcat([top_row, bottom_row])


# ========================================
# Initialization
# ========================================

device             = 'cuda' if torch.cuda.is_available() else 'cpu'
device_for_tracker = '0'   if torch.cuda.is_available() else 'cpu'

model = YOLO(YOLO_MODEL).to(device)
model_for_async_process = YOLO(YOLO_MODEL).to(device)
tracker = StrongSort(
    reid_weights=Path('osnet_x0_25_msmt17.pt'),
    device=device_for_tracker,
    half=True,
    max_cos_dist=0.15,
    max_iou_dist=0.9,
    n_init=3,
    min_conf=0.3,
)

cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
w   = TARGET_W
h   = TARGET_H
h   = h if h % 2 == 0 else h + 1

out_fps = fps / SKIP_FRAMES

ffmpeg_cmd = [
    'ffmpeg', '-y',
    '-f', 'rawvideo',
    '-vcodec', 'rawvideo',
    '-s', f'{w*2}x{h*2}',
    '-pix_fmt', 'bgr24',
    '-r', str(out_fps),
    '-i', '-',
    '-c:v', 'libx264',
    '-preset', 'ultrafast',
    '-crf', '28',
    '-pix_fmt', 'yuv420p',
    OUTPUT_H264_PATH
]
ffmpeg_proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)

print(f"Processing: {VIDEO_PATH} | Mode: 1 frame every {SKIP_FRAMES}")
print(f"Output resolution: {w*2}x{h*2} | Output FPS: {out_fps:.1f}")
print(f"Annotated video -> {OUTPUT_H264_PATH}")
print(f"Detection log   -> {OUTPUT_H264_PATH.replace('.mp4', '_detections.json')}")

# ========================================
# Main Loop
# ========================================

frame_idx = 0

people_id_remap                  : dict = {}  # new_id -> old_id (tracker sometimes reassigns IDs)
people_id_disappeared_already    : set  = set()
people_id_first_appeared_frame   : dict = {}
people_id_and_fg_path            : dict = {}
people_id_and_last_updated_fg    : dict = {}
people_id_and_init_box           : dict = {}
people_id_and_last_appeared_time : dict = {}
historical_frame                 : dict = {}

if DEBUG_DETAIL_OUTPUT:
    total_process_time_start = time.time()
    total_processed_frames   = 0

try:
    active_detections_for_visual = []
    active_visuals               = []

    while True:
        for _ in range(SKIP_FRAMES - 1):
            if not cap.grab(): break

        ret, frame = cap.read()
        if not ret: break

        frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_LINEAR)

        start_t = time.time()

        # Save historical frame snapshot
        if frame_idx % HISTORICAL_FRAME_SAVE_PER_FRAME == 0:
            historical_frame[frame_idx] = frame.copy()
        while len(historical_frame) > HISTORICAL_FRAME_MAX_SAVE:
            del historical_frame[min(historical_frame)]

        # --- Background subtraction ---
        if DEBUG_DETAIL_OUTPUT:
            debug_t = time.time()

        fg_fast = cntFast.apply(frame, learningRate=CNT_FAST_BOOST_LEARNING)
        fg_slow = cntSlow.apply(frame)
        fg_fast = process_raw_cnt_fg(fg_fast)
        fg_slow = process_raw_cnt_fg(fg_slow)

        if DEBUG_DETAIL_OUTPUT:
            print("Time for fg_fast and fg_slow: " + str((time.time() - debug_t) * 1000))

        # Ignore frames with massive sudden change (e.g. lighting flash, camera obstruction)
        if frame_idx > CNT_NO_IGNORE_IN_THE_BEGINNING_FRAME:
            total_pixels   = w * h
            changed_pixels = cv2.countNonZero(fg_fast)
            ratio          = changed_pixels / total_pixels
            if ratio > CNT_FAST_IGNORE_FRAME_IF_SUDDEN_CHANGE_EXCEEDS:
                frame_idx += SKIP_FRAMES
                print(f"[Warn] Frame {frame_idx} ignored: sudden change (fast) {ratio:.2%}")
                continue

        if frame_idx > CNT_NO_IGNORE_IN_THE_BEGINNING_FRAME:
            total_pixels   = w * h
            changed_pixels = cv2.countNonZero(fg_slow)
            ratio          = changed_pixels / total_pixels
            if ratio > CNT_SLOW_IGNORE_FRAME_IF_SUDDEN_CHANGE_EXCEEDS:
                frame_idx += SKIP_FRAMES
                print(f"[Warn] Frame {frame_idx} ignored: sudden change (slow) {ratio:.2%}")
                continue

        # --- YOLO inference + tracking ---
        if DEBUG_DETAIL_OUTPUT:
            debug_t = time.time()

        yolo_results, frame_info, vehicle_mask = process_single_frame(frame, frame_idx, model, tracker)

        if DEBUG_DETAIL_OUTPUT:
            print("Time for YOLO inference: " + str((time.time() - debug_t) * 1000))

        if DEBUG_DETAIL_OUTPUT:
            debug_t = time.time()

        # --- Update per-person walking paths ---
        for single_person_info in frame_info["persons"]:
            pid = single_person_info["id"]

            # ID remap: check if this new ID overlaps with an existing tracker tip
            if pid not in people_id_and_fg_path and pid not in people_id_disappeared_already and pid not in people_id_remap:
                x1, y1, x2, y2 = single_person_info["box"]
                best_match_pid   = None
                best_match_score = 0
                remap_candidates = []

                for existing_pid, tip_fg in people_id_and_last_updated_fg.items():
                    tip_pixels = cv2.countNonZero(tip_fg)
                    if tip_pixels == 0:
                        continue
                    box_mask = np.zeros_like(tip_fg)
                    cv2.rectangle(box_mask, (x1, y1), (x2, y2), 255, -1)
                    overlap_pixels    = cv2.countNonZero(cv2.bitwise_and(tip_fg, box_mask))
                    tip_in_box_ratio  = overlap_pixels / tip_pixels
                    box_area          = max((x2 - x1) * (y2 - y1), 1)
                    tip_in_box_fill   = overlap_pixels / box_area

                    passed = (
                        (tip_in_box_fill >= DICT_REMAP_TIP_IN_BOX_FILL and
                         tip_in_box_ratio >= DICT_REMAP_TIP_IN_BOX_FILL_MIN_RATIO) or
                        (tip_in_box_ratio >= DICT_REMAP_TIP_IN_BOX_RATIO)
                    )
                    candidate_info = {
                        "existing_pid": existing_pid,
                        "tip_pixels": tip_pixels,
                        "overlap_pixels": overlap_pixels,
                        "tip_in_box_ratio": tip_in_box_ratio,
                        "tip_in_box_fill": tip_in_box_fill,
                        "passed": passed,
                    }
                    if passed:
                        combined_score = (tip_in_box_ratio + tip_in_box_fill) / 2
                        candidate_info["combined_score"] = combined_score
                        if combined_score > best_match_score:
                            best_match_score = combined_score
                            best_match_pid   = existing_pid
                    else:
                        candidate_info["combined_score"] = 0
                    remap_candidates.append(candidate_info)

                if best_match_pid is not None:
                    people_id_remap[pid] = best_match_pid
                    print(f"  [ID Remap] new_id={pid} -> existing_id={best_match_pid}, score={best_match_score:.2%}")

            single_person_info["original_id"] = single_person_info["id"]
            if pid in people_id_remap:
                pid = people_id_remap[pid]
                single_person_info["id"] = pid

            if single_person_info["id"] in people_id_disappeared_already:
                continue

            people_id_and_last_appeared_time[single_person_info["id"]] = frame_idx

            if single_person_info["id"] not in people_id_and_fg_path:
                # First time seeing this ID — initialize their path from their bounding box
                people_id_and_init_box[single_person_info["id"]] = single_person_info["box"]
                init_fg = np.zeros((h, w), dtype=np.uint8)
                x1, y1, x2, y2 = single_person_info["box"]
                cv2.rectangle(init_fg, (x1, y1), (x2, y2), 255, -1)
                people_id_and_fg_path[single_person_info["id"]]        = init_fg
                people_id_and_last_updated_fg[single_person_info["id"]] = init_fg.copy()
                if single_person_info["id"] not in people_id_first_appeared_frame:
                    people_id_first_appeared_frame[single_person_info["id"]] = frame_idx

        if DEBUG_DETAIL_OUTPUT:
            print("Time to update tracks: " + str((time.time() - debug_t) * 1000))

        if DEBUG_DETAIL_OUTPUT:
            debug_t        = time.time()
            debug_walk_t   = time.time()

        # Combine all tips for whitelist computation
        all_tips = np.zeros((h, w), dtype=np.uint8)
        for pid in people_id_and_last_updated_fg:
            all_tips = cv2.bitwise_or(all_tips, people_id_and_last_updated_fg[pid])

        for pid, fg_path in list(people_id_and_fg_path.items()):
            if DEBUG_DETAIL_OUTPUT:
                debug_walk_single_t = time.time()

            # Whitelist = all other people's tips + vehicles
            whitelist = cv2.subtract(all_tips, people_id_and_last_updated_fg[pid])
            whitelist = cv2.bitwise_or(whitelist, vehicle_mask)

            new_walking_path_fg, new_last_updated_fg = update_walking_path(
                fg_path, people_id_and_last_updated_fg[pid], fg_fast, whitelist,
                init_box=people_id_and_init_box.get(pid)
            )

            people_id_and_fg_path[pid]        = new_walking_path_fg
            people_id_and_last_updated_fg[pid] = new_last_updated_fg

            # Clear the tip if the person has left the frame via an edge
            frames_missing    = frame_idx - people_id_and_last_appeared_time.get(pid, frame_idx)
            missing_long_enough = frames_missing >= UPDATE_WALKING_PATH_EDGE_CLEAR_MIN_MISSING_FRAMES * SKIP_FRAMES
            if missing_long_enough:
                init_box       = people_id_and_init_box.get(pid)
                path_large_enough = False
                if init_box is not None:
                    init_area  = max((init_box[2] - init_box[0]) * (init_box[3] - init_box[1]), 1)
                    path_area  = cv2.countNonZero(people_id_and_fg_path.get(pid, new_last_updated_fg))
                    path_large_enough = path_area >= init_area * UPDATE_WALKING_PATH_EDGE_CLEAR_MIN_PATH_RATIO
                if path_large_enough:
                    tip_coords = cv2.findNonZero(new_last_updated_fg)
                    if tip_coords is not None:
                        bx, by, bw, bh = cv2.boundingRect(tip_coords)
                        touches_edge = (
                            bx <= UPDATE_WALKING_PATH_TRACKER_RM_EDGE_MARGIN or
                            by <= UPDATE_WALKING_PATH_TRACKER_RM_EDGE_MARGIN or
                            bx + bw >= w - UPDATE_WALKING_PATH_TRACKER_RM_EDGE_MARGIN or
                            by + bh >= h - UPDATE_WALKING_PATH_TRACKER_RM_EDGE_MARGIN
                        )
                        if touches_edge:
                            people_id_and_last_updated_fg[pid] = np.zeros((h, w), dtype=np.uint8)
                            if DEBUG_DETAIL_OUTPUT:
                                print(f"  [edge_clear] pid={pid} tip at edge, cleared")

            if DEBUG_DETAIL_OUTPUT:
                walk_single_ms = (time.time() - debug_walk_single_t) * 1000
                tip_px  = cv2.countNonZero(people_id_and_last_updated_fg[pid])
                path_px = cv2.countNonZero(people_id_and_fg_path[pid])
                print(f"  [walk_path] pid={pid}, {walk_single_ms:.1f}ms, tip={tip_px}px, path={path_px}px")

        if DEBUG_DETAIL_OUTPUT:
            walk_total_ms = (time.time() - debug_walk_t) * 1000
            print(f"  [walk_path_total] {walk_total_ms:.1f}ms for {len(people_id_and_fg_path)} people")

        if len(people_id_and_fg_path) > 0:
            print(f"[main] frame={frame_idx}, tracking={len(people_id_and_fg_path)} people, fg_fast_pixels={cv2.countNonZero(fg_fast)}")

        # --- Dumping detection for people who have disappeared ---
        ids_to_cleanup = []
        for pid, last_frame in people_id_and_last_appeared_time.items():
            if pid in people_id_disappeared_already:
                continue
            elapsed_sec = (frame_idx - last_frame) / fps
            if elapsed_sec >= YOLO_VERIFY_AS_HUMAN_DISAPPEARED_AFTER_SECOND:
                ids_to_cleanup.append(pid)

        current_yolo_mask_combined = None

        for pid_detect in ids_to_cleanup:
            is_positive, res_fg_mask, visual_info = detect_anything_left(
                people_id_and_fg_path[pid_detect], fg_fast, fg_slow,
                people_id_and_init_box[pid_detect]
            )
            active_visuals.append(visual_info)

            if is_positive:
                merged_fg = np.zeros_like(res_fg_mask[0])
                for m in res_fg_mask:
                    merged_fg = cv2.bitwise_or(merged_fg, m)

                merge_kernel = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE,
                    (CHECK_AREA_MERGE_CLOSE_KERNEL_SIZE, CHECK_AREA_MERGE_CLOSE_KERNEL_SIZE)
                )
                merged_fg = cv2.morphologyEx(merged_fg, cv2.MORPH_CLOSE, merge_kernel)

                num_merged, labels_merged, stats_merged, _ = cv2.connectedComponentsWithStats(merged_fg, connectivity=8)
                merged_masks = []
                for i in range(1, num_merged):
                    comp = ((labels_merged == i).astype(np.uint8)) * 255
                    merged_masks.append(comp)

                # Find the best historical frame from before this person appeared
                historical_frame_start_number = people_id_first_appeared_frame[pid_detect]
                historical_frame_start_number = max(0, historical_frame_start_number - 10)
                while True:
                    if historical_frame_start_number in historical_frame:
                        historical_frame_start = historical_frame[historical_frame_start_number]
                        break
                    else:
                        historical_frame_start_number -= 1
                        if historical_frame_start_number < 0:
                            print("WARNING: historical frame lookup failed, using current frame as fallback")
                            historical_frame_start = frame
                            break

                def _check_trash_async(merged_masks, frame_snap, hist_snap, pid_label):
                    """Run trash confirmation in a background thread."""
                    yolo_mask_combined_local = None
                    for single_fg in merged_masks:
                        is_trash, yolo_wl, yolo_wl_hist = check_if_left_area_is_trash(
                            frame_snap, hist_snap, single_fg,
                            yolo_mask_combined_local, model_for_async_process
                        )
                        yolo_mask_combined_local = yolo_wl
                        if is_trash:
                            print(f"[async] pid={pid_label}: confirmed dumping detected")
                            active_detections_for_visual.extend(merged_masks)
                            # Log the detection event
                            log_detection_event(pid_label, frame_idx, fps, merged_masks)
                            break
                        else:
                            print(f"[async] pid={pid_label}: change detected but classified as object removal / false positive")

                t = threading.Thread(
                    target=_check_trash_async,
                    args=(merged_masks, frame.copy(), historical_frame_start.copy(), pid_detect),
                    daemon=True
                )
                t.start()

        # Clean up data for disappeared people
        for pid_del in ids_to_cleanup:
            people_id_disappeared_already.add(pid_del)
            for d in [people_id_and_fg_path, people_id_and_last_appeared_time,
                      people_id_and_init_box, people_id_and_last_updated_fg,
                      people_id_first_appeared_frame]:
                d.pop(pid_del, None)

        if DEBUG_DETAIL_OUTPUT:
            print("Time to process detections: " + str((time.time() - debug_t) * 1000))

        if DEBUG_DETAIL_OUTPUT:
            debug_t = time.time()

        # --- Visualization ---
        yolo_panel = yolo_results.plot()
        for p in frame_info["persons"]:
            x1, y1, x2, y2 = p["box"]
            tid         = p["id"]
            score       = p["score"]
            original_id = p.get("original_id", tid)
            cv2.rectangle(yolo_panel, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(yolo_panel, f"ID:{tid} {score:.2f}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            if original_id != tid:
                cv2.putText(yolo_panel, f"raw:{original_id}", (x1, y2 + 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)

        people_paths      = list(people_id_and_fg_path.values())
        active_moving_head = list(people_id_and_last_updated_fg.values())

        quad_frame = build_quad_view(
            frame, yolo_panel, fg_fast, fg_slow,
            active_detections_for_visual, people_paths,
            active_moving_head, active_visuals
        )

        if DEBUG_DETAIL_OUTPUT:
            print("Time to render quad view: " + str((time.time() - debug_t) * 1000))

        ffmpeg_proc.stdin.write(quad_frame.tobytes())

        if DEBUG_DETAIL_OUTPUT:
            elapsed = (time.time() - start_t) * 1000
            print(f"Frame {frame_idx} | {elapsed:.2f}ms")
        elif frame_idx % (SKIP_FRAMES * 10) == 0:
            elapsed = (time.time() - start_t) * 1000
            print(f"Frame {frame_idx} | {elapsed:.2f}ms")

        frame_idx += SKIP_FRAMES

        if DEBUG_DETAIL_OUTPUT:
            total_processed_frames += 1

finally:
    cap.release()
    ffmpeg_proc.stdin.close()
    ffmpeg_proc.wait()
    save_detection_log()  # Save JSON summary on exit
    print("Done!")

if DEBUG_DETAIL_OUTPUT:
    total_process_time = time.time() - total_process_time_start
    avg_ms      = (total_process_time / total_processed_frames) * 1000
    realtime_fps = 1.0 / (total_process_time / total_processed_frames)
    print(f"Processed {total_processed_frames} frames | avg {avg_ms:.2f}ms/frame | realtime would need {fps/SKIP_FRAMES:.1f}fps, achieved {realtime_fps:.1f}fps")