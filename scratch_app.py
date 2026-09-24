import os
# Configure CPU threads before imports to prevent thread thrashing
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import re
import cv2
import time
import uuid
import shutil
import threading
import torch
from datetime import datetime
from flask import Flask, render_template, Response, request, jsonify
from paddleocr import PaddleOCR
from ultralytics import YOLO

# Limit PyTorch CPU threads
torch.set_num_threads(1)

# Import report generator
from generate_report import create_inspection_report

app = Flask(__name__, template_folder='.', static_folder='.', static_url_path='')

# -------------------------------
# Configuration & Global States
# -------------------------------
MODEL_PATH = "best.pt"
YOLO_MODEL = None
OCR_ENGINE = None

# -------------------------------
# Detection Thresholds (adjust here)
# -------------------------------
YOLO_CONF_THRESHOLD = 0.15
  # Minimum YOLO confidence (0.0 - 1.0). Lower = more detections, Higher = stricter.

# Global lock for thread safety (using RLock to prevent self-deadlocks on nested acquisitions)
lock = threading.RLock()

# Global variables for processing state
stream_source = 0  # Default to Webcam
is_processing = False
video_cap = None

# Daily cycle count tracking
cycle_count = 1
last_date_str = datetime.now().strftime("%Y-%m-%d")

# Current cycle tracking states
current_cycle = {
    "status": "Awaiting camera connection",
    "cycle_number": "#001",
    "step1_status": "Pending",  # Waiting for Fuel Door
    "step2_status": "Pending",  # Front Side Captured
    "step3_status": "Pending",  # Back Side Captured
    "serial": "------",
    "confidence": "- -",
    "result": "Awaiting analysis...",
    "holes_count": 0,
    "defects": [],
    "vote_1": "- - - - - -",
    "vote_2": "- - - - - -",
    "vote_3": "- - - - - -",
    "instruction": "WAITING FOR PART",
    "instruction_color": "blue"
}

active_cycle_data = {
    "temp_folder": None,
    "front_path": None,
    "back_path": None,
    "serial_number": None,
    "ocr_confidence": 0.0,
    "processing_thread_active": False,
    "ocr_thread_active": False,
    "defects_detected": set(),
    "max_holes_detected": 0,
    "serial_votes": [],  # List of dicts: {"text": "123456", "confidence": 92.5}
    "back_frames_count": 0,
    "front_frames_count": 0,
    "serial_frames_count": 0,
    "frames_since_last_ocr_crop": 10,
    "state": "WAITING_FRONT",
    "front_box_center": None,
    "back_box_center": None,
    "remove_frames_count": 0,
    "no_panel_frames_count": 0,  # Tracks consecutive frames with sub-features but no front/back panel
    "ocr_start_time": None  # Timestamp when CHECKING_OCR state began (for timeout)
}

# Global placeholders for decoupled streaming speedup
latest_raw_frame = None
current_detections = []
latest_annotated_frame = None

# -------------------------------
# Initialization Helper
# -------------------------------
def init_models():
    global YOLO_MODEL, OCR_ENGINE
    if YOLO_MODEL is None:
        print("Loading YOLO Model...")
        if os.path.exists(MODEL_PATH):
            YOLO_MODEL = YOLO(MODEL_PATH)
        else:
            YOLO_MODEL = YOLO("yolov8n.pt")  # Fallback
    if OCR_ENGINE is None:
        print("Loading PaddleOCR Engine...")
        # PaddleOCR is CPU-only (paddlepaddle CPU build avoids CUDA conflict with torch)
        print("GPU Acceleration for OCR: Disabled (CPU build — YOLO still uses GPU)")
        OCR_ENGINE = PaddleOCR(
            use_angle_cls=False,  # Skip angle classifier for speed (we rotate manually)
            lang="en",
            use_gpu=False,        # paddlepaddle CPU build installed
            show_log=False
        )
    print("Models Initialized.")

# -------------------------------
# Asynchronous Background Processing
# -------------------------------
def run_ocr_on_crop(crop_rgb):
    """Try OCR on a crop in all 4 rotations. Returns (text, confidence) or (None, 0)."""
    rotations = [None, cv2.ROTATE_90_COUNTERCLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_CLOCKWISE]
    best_text = None
    best_conf = 0.0

    for rot in rotations:
        img = cv2.rotate(crop_rgb, rot) if rot is not None else crop_rgb
        try:
            # PaddleOCR 2.x API: returns [[[box, (text, conf)], ...]] per image
            raw = OCR_ENGINE.ocr(img, cls=False)
        except Exception as e:
            print(f"[OCR Error] Prediction failed on rotation {rot}: {e}")
            continue
        if not raw or raw[0] is None:
            continue
        for line in raw[0]:
            if line is None:
                continue
            text = line[1][0]   # (text, confidence) tuple
            conf = line[1][1]
            # Character substitutions for metallic/laser etched serials
            text_sub = text.strip().upper()
            text_sub = (text_sub.replace('O','0').replace('I','1').replace('L','1')
                        .replace('S','5').replace('Z','2').replace('B','8')
                        .replace('G','6').replace('T','7'))
            clean_text = re.sub(r"\D", "", text_sub)
            if len(clean_text) == 6 and float(conf) > best_conf:
                best_text = clean_text
                best_conf = float(conf)
        # If we already found a very high-confidence result, stop trying rotations
        if best_text and best_conf >= 0.88:
            break

    return best_text, best_conf * 100 if best_text else 0.0


# -------------------------------
# Asynchronous Background Processing
# -------------------------------
def process_ocr_async(crop_img, temp_folder, crop_num):
    """PaddleOCR processing running on a separate thread."""
    global current_cycle, active_cycle_data
    try:
        # Save crop for records
        crops_dir = os.path.join(temp_folder, "crops")
        os.makedirs(crops_dir, exist_ok=True)
        crop_path = os.path.join(crops_dir, f"serial_crop_{crop_num}.jpg")
        cv2.imwrite(crop_path, crop_img)
        print(f"[OCR Thread] Saved crop {crop_num}")

        # Convert BGR to RGB for PaddleOCR
        crop_rgb = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
        detected_serial, confidence = run_ocr_on_crop(crop_rgb)

        if detected_serial:
            with lock:
                active_cycle_data["serial_votes"].append({
                    "text": detected_serial,
                    "confidence": confidence
                })

                # Show result in UI immediately
                current_cycle["serial"] = detected_serial
                current_cycle["confidence"] = f"{confidence:.1f}%"

                votes = active_cycle_data["serial_votes"]
                if len(votes) >= 1:
                    current_cycle["vote_1"] = f"{votes[-1]['text']} ({votes[-1]['confidence']:.1f}%)"
                if len(votes) >= 2:
                    current_cycle["vote_2"] = f"{votes[-2]['text']} ({votes[-2]['confidence']:.1f}%)"
                if len(votes) >= 3:
                    current_cycle["vote_3"] = f"{votes[-3]['text']} ({votes[-3]['confidence']:.1f}%)"

                print(f"[OCR Thread] Crop {crop_num}: {detected_serial} ({confidence:.1f}%)")

                from collections import Counter
                texts = [v["text"] for v in votes]
                counter = Counter(texts)
                most_common_text, count = counter.most_common(1)[0]
                winning_votes = [v for v in votes if v["text"] == most_common_text]
                max_winning_conf = max(v["confidence"] for v in winning_votes)

                # Finalize if 2 matching votes OR any single read with >= 70% confidence
                if count >= 2 or max_winning_conf >= 70.0:
                    final_serial = most_common_text
                    final_conf = max_winning_conf  # use best confidence

                    active_cycle_data["serial_number"] = final_serial
                    active_cycle_data["ocr_confidence"] = final_conf
                    current_cycle["serial"] = final_serial
                    current_cycle["confidence"] = f"{final_conf:.2f}%"

                    if current_cycle["status"].startswith("Finished Cycle"):
                        current_cycle["status"] = "Finished Cycle for " + final_serial

                    current_cycle["vote_1"] = f"{final_serial} ({winning_votes[0]['confidence']:.1f}%)"
                    if len(winning_votes) >= 2:
                        current_cycle["vote_2"] = f"{winning_votes[1]['text']} ({winning_votes[1]['confidence']:.1f}%)"
                    if len(winning_votes) >= 3:
                        current_cycle["vote_3"] = f"{winning_votes[2]['text']} ({winning_votes[2]['confidence']:.1f}%)"

                    print(f"[OCR Thread] Finalized: {final_serial} ({final_conf:.1f}%)")
                    check_and_finalize_cycle()
        else:
            print(f"[OCR Thread] Crop {crop_num}: no valid 6-digit serial found")
    except Exception as e:
        print(f"[OCR Thread] Error crop {crop_num}: {e}")
    finally:
        with lock:
            active_cycle_data["ocr_thread_active"] = False

def finalize_report_and_rename():
    """Generates report and renames temp folder async to avoid thread dependencies."""
    global current_cycle, active_cycle_data, cycle_count, last_date_str
    try:
        with lock:
            active_cycle_data["finalize_started"] = True
            current_cycle["instruction"] = "CHECKING SERIAL..."
            current_cycle["instruction_color"] = "blue"
            current_cycle["status"] = "Reading Serial Number..."
        
        # Wait for OCR to finish reading serial (up to 5 seconds)
        ocr_wait_start = time.time()
        while time.time() - ocr_wait_start < 5.0:
            with lock:
                if active_cycle_data["serial_number"] is not None:
                    print(f"[Finalize] OCR ready: {active_cycle_data['serial_number']}")
                    break
                # Check if any votes came in while waiting
                if active_cycle_data["serial_votes"] and not active_cycle_data["ocr_thread_active"]:
                    best_vote = max(active_cycle_data["serial_votes"], key=lambda v: v["confidence"])
                    active_cycle_data["serial_number"] = best_vote["text"]
                    active_cycle_data["ocr_confidence"] = best_vote["confidence"]
                    current_cycle["serial"] = best_vote["text"]
                    current_cycle["confidence"] = f"{best_vote['confidence']:.1f}%"
                    print(f"[Finalize] Using best OCR vote: {best_vote['text']} ({best_vote['confidence']:.1f}%)")
                    break
            time.sleep(0.3)
        
        # Last resort: if OCR still has nothing after waiting
        with lock:
            if active_cycle_data["serial_number"] is None:
                if active_cycle_data["serial_votes"]:
                    best_vote = max(active_cycle_data["serial_votes"], key=lambda v: v["confidence"])
                    active_cycle_data["serial_number"] = best_vote["text"]
                    active_cycle_data["ocr_confidence"] = best_vote["confidence"]
                    current_cycle["serial"] = best_vote["text"]
                    current_cycle["confidence"] = f"{best_vote['confidence']:.1f}%"
                    print(f"[Finalize] Late OCR vote: {best_vote['text']} ({best_vote['confidence']:.1f}%)")
                else:
                    active_cycle_data["serial_number"] = "serial_missing"
                    active_cycle_data["ocr_confidence"] = 0.0
                    current_cycle["serial"] = "serial_missing"
                    current_cycle["confidence"] = "0.0%"
                    print("[Finalize] No OCR result after 5s wait, using serial_missing")
                    
        # Feature: Recheck if serial is missing
        is_missing = False
        with lock:
            if active_cycle_data["serial_number"] == "serial_missing":
                is_missing = True
                
        if is_missing:
            with lock:
                current_cycle["instruction"] = "RECHECKING SERIAL..."
                current_cycle["instruction_color"] = "red"
                current_cycle["status"] = "Confirming Missing Serial..."
                # Reset start time so fallback logic triggers again if needed
                active_cycle_data["ocr_start_time"] = time.time()
                active_cycle_data["serial_number"] = None
                
            # Wait another 5 seconds for recheck
            recheck_wait_start = time.time()
            while time.time() - recheck_wait_start < 5.0:
                with lock:
                    if active_cycle_data["serial_number"] is not None and active_cycle_data["serial_number"] != "serial_missing":
                        break
                    if active_cycle_data["serial_votes"] and not active_cycle_data["ocr_thread_active"]:
                        best_vote = max(active_cycle_data["serial_votes"], key=lambda v: v["confidence"])
                        active_cycle_data["serial_number"] = best_vote["text"]
                        active_cycle_data["ocr_confidence"] = best_vote["confidence"]
                        current_cycle["serial"] = best_vote["text"]
                        current_cycle["confidence"] = f"{best_vote['confidence']:.1f}%"
                        break
                time.sleep(0.3)
                
            with lock:
                if active_cycle_data["serial_number"] is None:
                    active_cycle_data["serial_number"] = "serial_missing"
                    active_cycle_data["ocr_confidence"] = 0.0
                    current_cycle["serial"] = "serial_missing"
                    current_cycle["confidence"] = "0.0%"
                    print("[Finalize] Recheck completed: Still serial_missing")
                else:
                    print(f"[Finalize] Recheck succeeded: {active_cycle_data['serial_number']}")

        # Now that we have the final serial number, instruct operator to remove the plate
        with lock:
            current_cycle["instruction"] = "REMOVE THE PLATE"
            current_cycle["instruction_color"] = "red"
            current_cycle["status"] = "Saving report..."
        
        temp_dir = active_cycle_data["temp_folder"]
        serial = active_cycle_data["serial_number"]
        front = active_cycle_data["front_path"]
        back = active_cycle_data["back_path"]
        conf = active_cycle_data["ocr_confidence"]
        
        # Read defects and verify holes
        with lock:
            defects = list(active_cycle_data["defects_detected"])
            if active_cycle_data["max_holes_detected"] < 2:
                defects.append("holes_missing")
            if serial == "serial_missing":
                defects.append("serial_missing")
            current_cycle["defects"] = defects
        status = "FAIL" if defects else "PASS"
        
        # Check folder structure and resolve if serial folder already exists
        today_str = datetime.now().strftime("%Y-%m-%d")
        day_dir = os.path.join("lid_data", today_str)
        os.makedirs(day_dir, exist_ok=True)
        
        timestamp_str = datetime.now().strftime("%H%M%S")
        
        if serial == "serial_missing":
            folder_name = f"serial_missing - {timestamp_str}"
        else:
            base_serial = serial
            base_path = os.path.join(day_dir, base_serial)
            if os.path.exists(base_path):
                # Same serial exists! Use timestamp for folder, images, and report
                folder_name = f"{base_serial}_{timestamp_str}"
            else:
                folder_name = base_serial

        print(f"[Finalize Thread] Building report for Serial: {serial} in {temp_dir} with status: {status}")

        # Path of the report within the temp folder matching the folder_name
        temp_report_path = os.path.join(temp_dir, f"{folder_name}.pdf")

        # Create report PDF
        create_inspection_report(
            serial_number=serial,
            model_name="CPIP Cam",
            status=status,
            date_str=datetime.now().strftime("%Y-%m-%d"),
            time_str=datetime.now().strftime("%H:%M:%S"),
            traceability_id=f"TRC{folder_name}",
            front_image_path=front,
            back_image_path=back,
            ocr_serial=serial,
            confidence=conf,
            output_path=temp_report_path
        )

        # Final destination path
        final_dir = os.path.join(day_dir, folder_name)
        if os.path.exists(final_dir):
            shutil.rmtree(final_dir)  # Clean if exists (unlikely with HHMMSS suffix)

        # Rename directory from temp to final folder name
        os.rename(temp_dir, final_dir)
        print(f"[Finalize Thread] Successfully moved and finalized folder to: {final_dir}")

        # Rename the images inside the final directory to match the target folder name
        try:
            old_front_path = os.path.join(final_dir, "front.jpg")
            new_front_path = os.path.join(final_dir, f"{folder_name}_front.jpg")
            if os.path.exists(old_front_path):
                os.rename(old_front_path, new_front_path)

            old_back_path = os.path.join(final_dir, "back.jpg")
            new_back_path = os.path.join(final_dir, f"{folder_name}_back.jpg")
            if os.path.exists(old_back_path):
                os.rename(old_back_path, new_back_path)
            
            print(f"[Finalize Thread] Renamed images to match folder name: {folder_name}")
        except Exception as rename_err:
            print(f"[Finalize Thread] Error renaming images: {rename_err}")

        with lock:
            # Check and increment cycle count daily
            if today_str != last_date_str:
                last_date_str = today_str
                cycle_count = 1
            else:
                cycle_count += 1

            current_cycle["result"] = status
            current_cycle["status"] = "Finished Cycle for " + serial

    except Exception as e:
        print(f"[Finalize Thread] Error: {e}")

def check_and_finalize_cycle():
    """Verifies if front and back have been captured, then triggers finalization."""
    global active_cycle_data
    if (active_cycle_data["front_path"] and 
        active_cycle_data["back_path"] and 
        not active_cycle_data["processing_thread_active"]):
        
        active_cycle_data["processing_thread_active"] = True
        threading.Thread(target=finalize_report_and_rename, daemon=True).start()

def reset_cycle_state():
    global current_cycle, active_cycle_data, cycle_count
    with lock:
        current_cycle["status"] = "Waiting for Fuel Door"
        current_cycle["cycle_number"] = f"#{cycle_count:03d}"
        current_cycle["step1_status"] = "Pending"
        current_cycle["step2_status"] = "Pending"
        current_cycle["step3_status"] = "Pending"
        current_cycle["serial"] = "------"
        current_cycle["confidence"] = "- -"
        current_cycle["result"] = "Awaiting analysis..."
        current_cycle["holes_count"] = 0
        current_cycle["defects"] = []
        current_cycle["vote_1"] = "- - - - - -"
        current_cycle["vote_2"] = "- - - - - -"
        current_cycle["vote_3"] = "- - - - - -"
        current_cycle["instruction"] = "WAITING FOR PART"
        current_cycle["instruction_color"] = "blue"
        
        active_cycle_data = {
            "temp_folder": None,
            "front_path": None,
            "back_path": None,
            "serial_number": None,
            "ocr_confidence": 0.0,
            "processing_thread_active": False,
            "ocr_thread_active": False,
            "defects_detected": set(),
            "max_holes_detected": 0,
            "serial_votes": [],
            "back_frames_count": 0,
            "front_frames_count": 0,
            "serial_frames_count": 0,
            "frames_since_last_ocr_crop": 10,
            "state": "WAITING_FRONT",
            "front_box_center": None,
            "back_box_center": None,
            "remove_frames_count": 0,
            "no_panel_frames_count": 0,
            "ocr_start_time": None
        }

# -------------------------------
# Background Video Processing Loop (RTSP/File Grabber & Annotator)
# -------------------------------
def video_processing_loop():
    global video_cap, is_processing, stream_source, latest_raw_frame, latest_annotated_frame, current_detections
    consecutive_failures = 0
    
    while True:
        if not is_processing or video_cap is None or not video_cap.isOpened():
            time.sleep(0.05)
            continue
            
        ret, frame = video_cap.read()
        if not ret:
            consecutive_failures += 1
            if consecutive_failures > 30:
                print("[Video Feed] Too many consecutive frame read failures. Pausing feed.")
                with lock:
                    is_processing = False
                continue
                
            # Loop video if source is a file
            if isinstance(stream_source, str) and not stream_source.startswith("rtsp"):
                print("[Video Feed] Loop reached: Re-opening video file.")
                with lock:
                    video_cap.release()
                    video_cap = cv2.VideoCapture(stream_source)
                consecutive_failures = 0
                time.sleep(0.03)
                continue
            else:
                time.sleep(0.05)
                continue
        
        consecutive_failures = 0
        
        # Save raw frame for YOLO thread
        with lock:
            latest_raw_frame = frame.copy()
            
        # Resize frame to standard 640px width first to speed up JPEG encoding and keep annotations crisp
        h_ann, w_ann = frame.shape[:2]
        if w_ann > 640:
            scale_ann = 640 / w_ann
            annotated_frame = cv2.resize(frame, (640, int(h_ann * scale_ann)))
        else:
            scale_ann = 1.0
            annotated_frame = frame.copy()
            
        with lock:
            dets = list(current_detections)
            
        for det in dets:
            # Scale box coordinates to the resized frame size
            x1 = int(det["box"][0] * scale_ann)
            y1 = int(det["box"][1] * scale_ann)
            x2 = int(det["box"][2] * scale_ann)
            y2 = int(det["box"][3] * scale_ann)
            
            class_name = det["class_name"]
            conf = det["conf"]
            
            # Color coding
            color = (0, 255, 0)
            if class_name in ["front", "circle_front"]: color = (0, 165, 255) # Orange in BGR
            elif class_name in ["back", "circle_back"]: color = (255, 0, 255) # Magenta
            elif class_name in ["serial", "serial_area"]: color = (255, 255, 0) # Cyan
            
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            label = f"{class_name}"
            
            # Solid background for text
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            text_x = x1
            text_y = max(y1 - 5, text_h + 5)
            cv2.rectangle(annotated_frame, (text_x, text_y - text_h - 4), (text_x + text_w, text_y + 2), color, -1)
            cv2.putText(annotated_frame, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        # Encode annotated frame to JPEG
        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        if ret:
            with lock:
                latest_annotated_frame = buffer.tobytes()
        
        # Speed throttle - read at true video FPS
        if isinstance(stream_source, str) and not stream_source.startswith("rtsp"):
            fps = video_cap.get(cv2.CAP_PROP_FPS) if video_cap else 30
            if fps <= 0:
                fps = 30
            time.sleep(1.0 / fps)
        else:
            time.sleep(0.001)

# -------------------------------
# Dedicated Background YOLO Worker Thread
# -------------------------------
def yolo_worker_loop():
    global YOLO_MODEL, latest_raw_frame, current_detections, active_cycle_data, current_cycle
    init_models()
    
    while True:
        if not is_processing or latest_raw_frame is None:
            time.sleep(0.01)
            continue
            
        # Grab copy of latest raw frame
        with lock:
            frame_to_process = latest_raw_frame.copy()
            state = active_cycle_data.get("state", "WAITING_FRONT")
            
        h_orig, w_orig = frame_to_process.shape[:2]
        
        # Resize frame for faster YOLO inference (640 is optimal speed/accuracy trade-off)
        scale = 1.0
        if w_orig > 640:
            scale = 640 / w_orig
            frame_resized = cv2.resize(frame_to_process, (640, int(h_orig * scale)))
        else:
            frame_resized = frame_to_process.copy()
            
        try:
            results = YOLO_MODEL(frame_resized, verbose=False, conf=YOLO_CONF_THRESHOLD)
            new_detections = []
            frame_holes = 0
            frame_defects = []
            
            # Frame with annotations for saving
            annotated_frame = frame_to_process.copy()
            has_front_detected = False
            has_back_detected = False
            front_box = None
            back_box = None
            
            # Track sub-feature detections for fallback (circular shapes that don't detect front/back)
            sub_feature_boxes = []  # Collect all non-front/back bounding boxes
            has_serial_detected = False
            has_holes_detected = False
            has_defect_detected = False
            
            if results:
                boxes = results[0].boxes
                names = results[0].names
                
                for box in boxes:
                    cls_id = int(box.cls[0].cpu().item())
                    if names[cls_id].lower() in ["front", "circle_front"]:
                        has_front_detected = True
                    elif names[cls_id].lower() in ["back", "circle_back"]:
                        has_back_detected = True
                
                for box in boxes:
                    xyxy_resized = box.xyxy[0].cpu().numpy().astype(int)
                    cls_id = int(box.cls[0].cpu().item())
                    class_name = names[cls_id].lower()
                    
                    if class_name in ["bulge", "dent"]:
                        continue
                        
                    conf = float(box.conf[0].cpu().item())
                    
                    # Map coordinates back to original frame size
                    x1 = int(xyxy_resized[0] / scale)
                    y1 = int(xyxy_resized[1] / scale)
                    x2 = int(xyxy_resized[2] / scale)
                    y2 = int(xyxy_resized[3] / scale)
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w_orig, x2), min(h_orig, y2)
                    
                    new_detections.append({
                        "box": [x1, y1, x2, y2],
                        "class_name": class_name,
                        "conf": conf
                    })
                    
                    # Draw annotations for saved images
                    color = (0, 255, 0)
                    if class_name in ["front", "circle_front"]: color = (0, 165, 255)
                    elif class_name in ["back", "circle_back"]: color = (255, 0, 255)
                    elif class_name in ["serial", "serial_area"]: color = (255, 255, 0)
                    
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 3)
                    label = f"{class_name}"
                    (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                    text_x = x1
                    text_y = max(y1 - 10, text_h + 10)
                    cv2.rectangle(annotated_frame, (text_x, text_y - text_h - 4), (text_x + text_w, text_y + 2), color, -1)
                    cv2.putText(annotated_frame, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
                    
                    # Track sub-features for fallback
                    if class_name in ["holes", "serial", "serial_area"]:
                        sub_feature_boxes.append((x1, y1, x2, y2))
                        if class_name in ["serial", "serial_area"]:
                            has_serial_detected = True
                        elif class_name == "holes":
                            has_holes_detected = True
                    
                    if class_name == "holes":
                        # Count holes when back is detected OR in fallback mode (no panel detected)
                        if has_back_detected or (not has_front_detected and not has_back_detected):
                            frame_holes += 1
                    elif class_name in ["front", "circle_front"]:
                        front_box = (x1, y1, x2, y2)
                    elif class_name in ["back", "circle_back"]:
                        back_box = (x1, y1, x2, y2)
                        
                    # OCR logic (runs asynchronously)
                    if class_name in ["serial", "serial_area"] and active_cycle_data["serial_number"] is None and conf >= 0.40:
                        with lock:
                            if active_cycle_data["ocr_start_time"] is None:
                                active_cycle_data["ocr_start_time"] = time.time()
                            time_elapsed = time.time() - active_cycle_data["ocr_start_time"]

                        is_ocr_target = False
                        # Try 'serial' class for the first 3.5 seconds
                        if class_name == "serial" and time_elapsed <= 3.5:
                            is_ocr_target = True
                        # If 3.5 seconds pass without a successful read, fallback to 'serial_area'
                        elif class_name == "serial_area" and time_elapsed > 3.5:
                            is_ocr_target = True
                        # Safety fallback: if we are past 3.5s but no serial_area was detected in this frame, still allow serial
                        elif class_name == "serial" and time_elapsed > 3.5 and not any(d["class_name"] == "serial_area" for d in new_detections):
                            is_ocr_target = True

                        if is_ocr_target:
                            with lock:
                                should_start_ocr = not active_cycle_data["ocr_thread_active"]

                            if should_start_ocr:
                                with lock:
                                    active_cycle_data["ocr_thread_active"] = True

                                padding = 15
                                x1_pad = max(0, x1 - padding)
                                y1_pad = max(0, y1 - padding)
                                x2_pad = min(w_orig, x2 + padding)
                                y2_pad = min(h_orig, y2 + padding)

                                crop = frame_to_process[y1_pad:y2_pad, x1_pad:x2_pad]

                                h_crop, w_crop = crop.shape[:2]
                                if h_crop > 0 and h_crop < 80:
                                    scale_factor = 3.0 if h_crop < 40 else 2.0
                                    crop = cv2.resize(crop, (0, 0), fx=scale_factor, fy=scale_factor,
                                                      interpolation=cv2.INTER_CUBIC)

                                next_crop_num = len(active_cycle_data["serial_votes"]) + 1
                                threading.Thread(
                                    target=process_ocr_async,
                                    args=(crop, active_cycle_data["temp_folder"], next_crop_num),
                                    daemon=True
                                ).start()
                
                
            with lock:
                current_detections = new_detections
                needs_reset = False
                
                # Strict State Machine Logic
                if state == "WAITING_FRONT":
                    current_cycle["status"] = "Waiting for Front Panel"
                    current_cycle["instruction"] = "PLACE FUEL DOOR (FRONT)"
                    current_cycle["instruction_color"] = "blue"
                    
                    if front_box and not has_back_detected:
                        if active_cycle_data["temp_folder"] is None:
                            today_str = datetime.now().strftime("%Y-%m-%d")
                            timestamp = datetime.now().strftime("%H%M%S")
                            temp_path = os.path.join("lid_data", today_str, f"temp_capture_{timestamp}")
                            os.makedirs(temp_path, exist_ok=True)
                            active_cycle_data["temp_folder"] = temp_path
                            current_cycle["step1_status"] = "OK"
                            
                        # Anti-shake distance check
                        cx = (front_box[0] + front_box[2]) / 2
                        cy = (front_box[1] + front_box[3]) / 2
                        prev_cx, prev_cy = active_cycle_data.get("front_box_center") or (cx, cy)
                        dist = ((cx - prev_cx)**2 + (cy - prev_cy)**2)**0.5
                        
                        if dist > 10:
                            active_cycle_data["front_frames_count"] = 0
                            
                        active_cycle_data["front_box_center"] = (cx, cy)
                        active_cycle_data["front_frames_count"] += 1
                        
                        if active_cycle_data["front_frames_count"] >= 2:
                            # Save full raw feed image without annotations and without cropping
                            full_image = frame_to_process.copy()
                            front_file = os.path.join(active_cycle_data["temp_folder"], "front.jpg")
                            cv2.imwrite(front_file, full_image, [cv2.IMWRITE_JPEG_QUALITY, 95])
                            
                            active_cycle_data["front_path"] = front_file
                            current_cycle["step2_status"] = "OK"
                            active_cycle_data["state"] = "WAITING_BACK"
                            for d in frame_defects:
                                active_cycle_data["defects_detected"].add(d)
                    else:
                        active_cycle_data["front_frames_count"] = 0
                elif state == "WAITING_BACK":
                    current_cycle["status"] = "Waiting for Back Panel"
                    current_cycle["instruction"] = "FLIP TO BACK SIDE"
                    current_cycle["instruction_color"] = "blue"
                    
                    is_back_visible = (back_box is not None) or has_serial_detected
                    
                    if is_back_visible and not has_front_detected:
                        # Determine a center for movement tracking
                        if back_box:
                            cx = (back_box[0] + back_box[2]) / 2
                            cy = (back_box[1] + back_box[3]) / 2
                        elif len(sub_feature_boxes) > 0:
                            cx = sum([b[0] + b[2] for b in sub_feature_boxes]) / (2 * len(sub_feature_boxes))
                            cy = sum([b[1] + b[3] for b in sub_feature_boxes]) / (2 * len(sub_feature_boxes))
                        else:
                            cx, cy = w_orig / 2, h_orig / 2
                            
                        prev_cx, prev_cy = active_cycle_data.get("back_box_center") or (cx, cy)
                        dist = ((cx - prev_cx)**2 + (cy - prev_cy)**2)**0.5
                        
                        if dist > 50:
                            active_cycle_data["back_frames_count"] = 0
                            
                        active_cycle_data["back_box_center"] = (cx, cy)
                        active_cycle_data["back_frames_count"] += 1
                        
                        if active_cycle_data["back_frames_count"] >= 2:
                            if has_serial_detected:
                                # Save full annotated image without cropping
                                full_raw = frame_to_process.copy()
                                back_file = os.path.join(active_cycle_data["temp_folder"], "back.jpg")
                                cv2.imwrite(back_file, full_raw, [cv2.IMWRITE_JPEG_QUALITY, 95])
                                active_cycle_data["back_path"] = back_file
                                
                                current_cycle["step3_status"] = "OK"
                                for d in frame_defects:
                                    active_cycle_data["defects_detected"].add(d)
                                
                                # Go straight to finalization — OCR will be waited on in background thread
                                active_cycle_data["state"] = "WAITING_REMOVE"
                                check_and_finalize_cycle()
                            else:
                                current_cycle["instruction"] = "DETECTING SERIAL NUMBER..."
                                active_cycle_data["back_frames_count"] = 1 # Keep it below threshold until serial is seen
                    else:
                        active_cycle_data["back_frames_count"] = 0
                        
                elif state == "WAITING_REMOVE":
                    # Check for ANY detection (panel or sub-features from circular shapes)
                    any_lid_visible = has_front_detected or has_back_detected or len(sub_feature_boxes) > 0
                    if not any_lid_visible:
                        active_cycle_data["remove_frames_count"] += 1
                        if active_cycle_data["remove_frames_count"] >= 3:
                            needs_reset = True
                    else:
                        active_cycle_data["remove_frames_count"] = 0
                        
                # Update UI data outside state transitions
                if state in ["WAITING_FRONT", "WAITING_BACK", "WAITING_REMOVE"]:
                    current_cycle["holes_count"] = min(2, max(current_cycle["holes_count"], frame_holes))
                    # Continuously accumulate defects from every frame (not just at capture time)
                    for d in frame_defects:
                        active_cycle_data["defects_detected"].add(d)
                    current_cycle["defects"] = list(active_cycle_data["defects_detected"])
                    if active_cycle_data["temp_folder"] is not None:
                        active_cycle_data["max_holes_detected"] = min(2, max(active_cycle_data["max_holes_detected"], frame_holes))

            if needs_reset:
                reset_cycle_state()
                
        except Exception as e:
            print(f"[YOLO Thread Inference Error] {e}")
            
        time.sleep(0.01)

# -------------------------------
# Core Frame Generator Loop (Clients stream here)
# -------------------------------
def gen_frames():
    global latest_annotated_frame, is_processing
    last_yielded_frame = None
    
    while True:
        if not is_processing or latest_annotated_frame is None:
            time.sleep(0.03)
            continue
            
        frame_bytes = latest_annotated_frame
        if frame_bytes != last_yielded_frame:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            last_yielded_frame = frame_bytes
        else:
            time.sleep(0.01)

# -------------------------------
# API Endpoints
# -------------------------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/connect_camera', methods=['POST'])
def connect_camera():
    global video_cap, is_processing, stream_source
    data = request.json
    ip = data.get('ip')
    port = data.get('port')
    user = data.get('user')
    pwd = data.get('pwd')
    path = data.get('path')
    
    rtsp_url = f"rtsp://{user}:{pwd}@{ip}:{port}{path}"
    print(f"[Camera Endpoint] Connecting to RTSP Camera: rtsp://{user}:***@{ip}:{port}{path}")
    
    with lock:
        is_processing = False
        if video_cap:
            video_cap.release()
            print("[Camera Endpoint] Released previous video stream.")
            
        stream_source = rtsp_url
        video_cap = cv2.VideoCapture(stream_source)
        is_processing = True
        reset_cycle_state()
        print(f"[Camera Endpoint] Loaded RTSP stream. is_processing set to True.")
        
    return jsonify({"status": "success", "message": "Camera stream connected"})

@app.route('/upload_video', methods=['POST'])
def upload_video():
    global video_cap, is_processing, stream_source
    if 'video' not in request.files:
        print("[Upload Endpoint] Error: No video file in request.")
        return jsonify({"status": "error", "message": "No video file provided"}), 400
        
    file = request.files['video']
    print(f"[Upload Endpoint] File received: {file.filename}")
    temp_video_path = os.path.join("lid_data", "temp_uploaded_video.mp4")
    os.makedirs("lid_data", exist_ok=True)
    file.save(temp_video_path)
    print(f"[Upload Endpoint] File successfully saved to {temp_video_path}")
    
    with lock:
        is_processing = False
        if video_cap:
            video_cap.release()
            print("[Upload Endpoint] Released previous video stream.")
            
        stream_source = temp_video_path
        video_cap = cv2.VideoCapture(stream_source)
        is_processing = True
        reset_cycle_state()
        print(f"[Upload Endpoint] Loaded stream_source: '{stream_source}'. is_processing set to True.")
        
    return jsonify({"status": "success", "message": "Video uploaded and stream started"})

@app.route('/status')
def status():
    return jsonify(current_cycle)

if __name__ == '__main__':
    # Initialize workspace folders
    os.makedirs("lid_data", exist_ok=True)
    init_models()
    
    # Start the video processing background thread
    processing_thread = threading.Thread(target=video_processing_loop, daemon=True)
    processing_thread.start()
    
    # Start the YOLO inference worker thread
    yolo_thread = threading.Thread(target=yolo_worker_loop, daemon=True)
    yolo_thread.start()
    
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
