import os
# Configure CPU threads before imports to prevent PyTorch/PaddlePaddle thread thrashing
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
import atexit
import sys
import ctypes
import numpy as np
import yaml
import logging
from logging.handlers import RotatingFileHandler

# --- Initialization: Config & Logging ---
os.makedirs("logs", exist_ok=True)
log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
log_file = "logs/ssp_app.log"
file_handler = RotatingFileHandler(log_file, maxBytes=10*1024*1024, backupCount=5)
file_handler.setFormatter(log_formatter)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(log_formatter)

logger = logging.getLogger("SSP_App")
logger.setLevel(logging.INFO)
logger.addHandler(file_handler)
logger.addHandler(console_handler)

class StreamToLogger(object):
    def __init__(self, logger, log_level=logging.INFO):
        self.logger = logger
        self.log_level = log_level
        self.linebuf = ''
    def write(self, buf):
        for line in buf.rstrip().splitlines():
            self.logger.log(self.log_level, line.rstrip())
    def flush(self):
        pass

sys.stdout = StreamToLogger(logger, logging.INFO)
sys.stderr = StreamToLogger(logger, logging.ERROR)

# Load Configuration
CONFIG_PATH = "config.yaml"
if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH, "r") as f:
        APP_CONFIG = yaml.safe_load(f) or {}
    logger.info("Loaded configuration from config.yaml")
else:
    APP_CONFIG = {}
    logger.warning("config.yaml not found, using default values.")

# Def config defaults
OCR_CONF_THRESHOLD = APP_CONFIG.get("ai", {}).get("ocr_confidence_threshold", 0.20)
OCR_FALLBACK_TIMEOUT = APP_CONFIG.get("ai", {}).get("ocr_fallback_timeout", 3.5)
MODEL_PATH = APP_CONFIG.get("ai", {}).get("model_path", "ssp_yolov8-seg.pt")
RETENTION_DAYS = APP_CONFIG.get("storage", {}).get("retention_days", 30)


# --- IKapC SDK Integration ---
IKapLib_dir = r"C:\Program Files\I-TEK OptoElectronics\IKapLibrary\Examples\Python\IKapLib"
if IKapLib_dir not in sys.path:
    sys.path.append(IKapLib_dir)

try:
    import IKapC
    import IKapCDef
    sdk_available = True
except ImportError:
    print("IKapC SDK not found!")
    sdk_available = False
import torch
import functools
_orig_load = torch.load
def _patched_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _orig_load(*args, **kwargs)
torch.load = _patched_load
import torch
from datetime import datetime
from flask import Flask, render_template, Response, request, jsonify
import logging
werkzeug_log = logging.getLogger('werkzeug')
werkzeug_log.setLevel(logging.ERROR)
import urllib.request as _urllib_req
import urllib.error
import base64 as _base64
import json as _json

class _PaddleOCRClient:
    """HTTP client that calls the paddleocr_server.py microservice (py -3.12)."""
    PADDLE_URL = "http://127.0.0.1:5001/ocr"

    def __init__(self, *args, **kwargs):
        pass

    def predict_crop(self, img_bgr):
        """Send a BGR crop to the PaddleOCR server, returns list of {rec_text, rec_score}."""
        try:
            _, buf = cv2.imencode('.png', img_bgr)
            b64 = _base64.b64encode(buf).decode()
            payload = _json.dumps({"image": b64}).encode()
            req = _urllib_req.Request(self.PADDLE_URL, data=payload,
                                      headers={'Content-Type': 'application/json'})
            resp = _urllib_req.urlopen(req, timeout=15)  # Increased to 5 to give back-side crops more time
            raw_data = resp.read()
            data = _json.loads(raw_data)
            results = data.get("results", [])
            
            print(f"\n[OCR RESPONSE]")
            print(f"HTTP status = {resp.status}")
            print(f"raw response = {raw_data.decode('utf-8', errors='ignore')}")
            print(f"recognized texts = {[r.get('rec_text') for r in results]}\n")
            
            if results:
                return results
            else:
                print(f"[OCR ERROR] PaddleOCR returned no usable text")
                return []
        except urllib.error.HTTPError as e:
            try:
                err_body = e.read().decode('utf-8')
            except Exception:
                err_body = str(e)
            logger.info(f"raw response = {e} - Body: {err_body}")
            logger.info(f"recognized texts = []")
            logger.error(f"[OCR ERROR] PaddleOCR request failed: {e} - {err_body}")
            return []
        except Exception as e:
            logger.info(f"raw response = {e}")
            logger.info(f"recognized texts = []")
            logger.error(f"[OCR ERROR] PaddleOCR request failed: {e}")
            return []

PaddleOCR = _PaddleOCRClient

from ultralytics import YOLO

# Limit PyTorch CPU threads
torch.set_num_threads(1)

# Import report generator
from generate_report import create_inspection_report

app = Flask(__name__, template_folder='.', static_folder='.', static_url_path='')

# -------------------------------
# Configuration & Global States
# -------------------------------

YOLO_MODEL = None
OCR_ENGINE = None

# -------------------------------
# Detection Thresholds (adjust here)
# -------------------------------
YOLO_CONF_THRESHOLD = APP_CONFIG.get("ai", {}).get("default_yolo_confidence", 0.15)
CLASS_CONF_THRESHOLDS = APP_CONFIG.get("ai", {}).get("class_confidences", {})
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
    "step3_status": "Pending",  # Lock Striker Captured
    "step4_status": "Pending",  # Back Side Captured
    "serial": "------",
    "confidence": "- -",
    "result": "Awaiting analysis...",
    "holes_count": 0,
    "rod_count": 0,
    "striker_count": 0,
    "back_hook_count": 0,
    "lock_striker_count": 0,
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
    "max_rod_detected": 0,
    "max_striker_detected": 0,
    "max_back_hook_detected": 0,
    "max_lock_striker_detected": 0,
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
    "ocr_start_time": None,  # Timestamp when CHECKING_OCR state began (for timeout)
    "front_type": None,  # "standard" if 'front' detected, "circle" if 'circle_front' detected
    "defect_frame_path": None  # Path to annotated frame saved when first defect is detected on front
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
        # Check for GPU for YOLO logging, but force PaddleOCR to CPU
        has_gpu = torch.cuda.is_available()
        print(f"System GPU Acceleration Available (YOLO): {has_gpu}")
        OCR_ENGINE = PaddleOCR(
            use_textline_orientation=False,
            lang="en",
            use_gpu=False
        )
    print("Models Initialized.")

# -------------------------------
# Asynchronous Background Processing
# -------------------------------
def has_part(roi_bgr, blue_thresh, metal_min_ratio, laplacian_min):
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    h, w = hsv.shape[:2]
    total = h * w
    if total == 0:
        return False

    # Blue foam tray range - widened to catch darker shadows
    blue_lower = np.array([90, 40, 20])
    blue_upper = np.array([140, 255, 255])
    blue_mask = cv2.inRange(hsv, blue_lower, blue_upper)
    blue_ratio = cv2.countNonZero(blue_mask) / total

    # Metallic gray part range - updated to include bright white specular highlights
    gray_lower = np.array([0, 0, 50])
    gray_upper = np.array([180, 60, 255])
    gray_mask = cv2.inRange(hsv, gray_lower, gray_upper)
    gray_ratio = cv2.countNonZero(gray_mask) / total

    # If the region has enough gray/white, it is definitely the metal part.
    if gray_ratio >= metal_min_ratio:
        return True

    # If it lacks gray/white and is predominantly blue, it is the empty tray.
    if blue_ratio > blue_thresh:
        return False

    # Default to True to avoid incorrectly filtering out a part due to weird lighting
    return True

def run_ocr_on_crop(crop_rgb):
    """Try OCR on a crop in all 4 rotations. Returns (text, confidence) or (None, 0)."""
    rotations = [None, cv2.ROTATE_90_COUNTERCLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_CLOCKWISE]
    best_text = None
    best_conf = 0.0

    for rot in rotations:
        img = cv2.rotate(crop_rgb, rot) if rot is not None else crop_rgb
        try:
            results = OCR_ENGINE.predict_crop(img)
        except Exception as e:
            print(f"[OCR Error] Prediction failed on rotation {rot}: {e}")
            continue
        
        if not results:
            continue
            
        # Combine all detected text lines from the crop
        combined_text = " ".join([res.get("rec_text", "") for res in results])
        avg_conf = sum([res.get("rec_score", 0.0) for res in results]) / len(results)
        
        # Keep only alphanumeric characters for the serial number
        text_sub = combined_text.strip().upper()
        clean_text = re.sub(r"[^A-Z0-9]", "", text_sub)
        
        print(f"\n[OCR VALIDATION]")
        print(f"raw text = {combined_text}")
        print(f"cleaned text = {clean_text}")
        
        # 1st 6 digits (date) + 3 digits + 1 letter A/B/C + 4 digits (time)
        match = re.search(r"(\d{9}[A-C]\d{4})", clean_text)
        
        if not match:
            print(f"accepted/rejected = rejected")
            print(f"reason = Did not match format: 6 digits (date) + 3 digits + 1 letter (A/B/C) + 4 digits (time)\n")
        else:
            clean_text = match.group(1)
            if float(avg_conf) > best_conf:
                print(f"accepted/rejected = accepted")
                print(f"reason = Valid format and higher confidence ({avg_conf} > {best_conf})\n")
                best_text = clean_text
                best_conf = float(avg_conf)
            else:
                print(f"accepted/rejected = rejected")
                print(f"reason = Valid format but confidence ({avg_conf}) not higher than best ({best_conf})\n")
                
        # If we already found a very high-confidence result, stop trying rotations
        if best_text and best_conf >= 0.88:
            break

    if not best_text:
        print(f"[OCR ERROR] OCR text rejected by serial validation")

    return best_text, best_conf * 100 if best_text else 0.0
# -------------------------------
# Asynchronous Background Processing
# -------------------------------
def process_ocr_async(crop_img, temp_folder, crop_num, c_data):
    """PaddleOCR processing running on a separate thread."""
    global current_cycle, active_cycle_data
    try:
        # Save crop for records
        if temp_folder is not None:
            crops_dir = os.path.join(temp_folder, "crops")
            os.makedirs(crops_dir, exist_ok=True)
            crop_path = os.path.join(crops_dir, f"serial_crop_{crop_num}.jpg")
            cv2.imwrite(crop_path, crop_img)
            logger.info(f"[OCR Thread] Saved crop {crop_num}")
        else:
            logger.info(f"[OCR Thread] Skipping crop {crop_num} save (temp_folder is None)")

        # The OCR client's imencode expects BGR, so do not convert to RGB
        detected_serial, confidence = run_ocr_on_crop(crop_img)

        if detected_serial:
            with lock:
                c_data["serial_votes"].append({
                    "text": detected_serial,
                    "confidence": confidence
                })
                
                logger.info(f"\n[OCR VOTE]")
                logger.info(f"text = {detected_serial}")
                logger.info(f"confidence = {confidence:.1f}%")
                logger.info(f"vote count = {len(c_data['serial_votes'])}\n")

                is_active = (c_data is active_cycle_data)

                # Show result in UI immediately if active
                if is_active:
                    ds_fmt = f"{detected_serial[:6]} {detected_serial[6:10]} {detected_serial[10:]}"
                    current_cycle["serial"] = ds_fmt
                    current_cycle["confidence"] = f"{confidence:.1f}%"

                votes = c_data["serial_votes"]
                if len(votes) >= 1 and is_active:
                    current_cycle["vote_1"] = f"{votes[-1]['text']} ({votes[-1]['confidence']:.1f}%)"
                if len(votes) >= 2 and is_active:
                    current_cycle["vote_2"] = f"{votes[-2]['text']} ({votes[-2]['confidence']:.1f}%)"
                if len(votes) >= 3 and is_active:
                    current_cycle["vote_3"] = f"{votes[-3]['text']} ({votes[-3]['confidence']:.1f}%)"

                logger.info(f"[OCR Thread] Crop {crop_num}: {detected_serial} ({confidence:.1f}%)")

                from collections import Counter
                texts = [v["text"] for v in votes]
                counter = Counter(texts)
                most_common_text, count = counter.most_common(1)[0]
                winning_votes = [v for v in votes if v["text"] == most_common_text]
                max_winning_conf = max(v["confidence"] for v in winning_votes)

                # Finalize if 2 matching votes OR any single read with >= 70% confidence
                if count >= 2 or max_winning_conf >= 20.0:
                    final_serial = most_common_text
                    final_conf = max_winning_conf  # use best confidence

                    c_data["serial_number"] = final_serial
                    c_data["ocr_confidence"] = final_conf
                    
                    logger.info(f"\n[OCR FINAL]")
                    logger.info(f"serial_number = {final_serial}")
                    logger.info(f"confidence = {final_conf:.1f}%\n")
                    
                    if is_active:
                        # Format as: YYMMDD NNNA HHMM
                        fs_fmt = f"{final_serial[:6]} {final_serial[6:10]} {final_serial[10:]}"
                        
                        current_cycle["serial"] = fs_fmt
                        current_cycle["confidence"] = f"{final_conf:.2f}%"
                        if current_cycle["status"].startswith("Finished Cycle"):
                            current_cycle["status"] = "Finished Cycle for " + fs_fmt

                        current_cycle["vote_1"] = f"{final_serial} ({winning_votes[0]['confidence']:.1f}%)"
                        if len(winning_votes) >= 2:
                            current_cycle["vote_2"] = f"{winning_votes[1]['text']} ({winning_votes[1]['confidence']:.1f}%)"
                        if len(winning_votes) >= 3:
                            current_cycle["vote_3"] = f"{winning_votes[2]['text']} ({winning_votes[2]['confidence']:.1f}%)"

                    logger.info(f"[OCR Thread] Finalized: {final_serial} ({final_conf:.1f}%)")
                    check_and_finalize_cycle(c_data)
        else:
            logger.info(f"[OCR Thread] Crop {crop_num}: no valid 14-16 character alphanumeric serial found")
    except Exception as e:
        logger.error(f"[OCR Thread] Error crop {crop_num}: {e}")
    finally:
        with lock:
            c_data["ocr_thread_active"] = False

def finalize_report_and_rename(c_data):
    """Generates report and renames temp folder async to avoid thread dependencies."""
    global current_cycle, active_cycle_data, cycle_count, last_date_str
    try:
        with lock:
            c_data["finalize_started"] = True
            if c_data is active_cycle_data:
                current_cycle["instruction"] = "CHECKING SERIAL..."
                current_cycle["instruction_color"] = "blue"
                current_cycle["status"] = "Reading Serial Number..."
        
        # Wait for OCR to finish reading serial (up to 15 seconds)
        ocr_wait_start = time.time()
        while time.time() - ocr_wait_start < 6.0:
            with lock:
                if c_data["serial_number"] is not None:
                    logger.info(f"[Finalize] OCR ready: {c_data['serial_number']}")
                    break
                # Check if any votes came in while waiting
                if c_data["serial_votes"] and not c_data["ocr_thread_active"]:
                    best_vote = max(c_data["serial_votes"], key=lambda v: v["confidence"])
                    c_data["serial_number"] = best_vote["text"]
                    c_data["ocr_confidence"] = best_vote["confidence"]
                    if c_data is active_cycle_data:
                        current_cycle["serial"] = best_vote["text"]
                        current_cycle["confidence"] = f"{best_vote['confidence']:.1f}%"
                    logger.info(f"[Finalize] Using best OCR vote: {best_vote['text']} ({best_vote['confidence']:.1f}%)")
                    break
            time.sleep(0.3)
        
        # Last resort: if OCR still has nothing after waiting
        with lock:
            if c_data["serial_number"] is None:
                if c_data["serial_votes"]:
                    best_vote = max(c_data["serial_votes"], key=lambda v: v["confidence"])
                    c_data["serial_number"] = best_vote["text"]
                    c_data["ocr_confidence"] = best_vote["confidence"]
                    if c_data is active_cycle_data:
                        current_cycle["serial"] = best_vote["text"]
                        current_cycle["confidence"] = f"{best_vote['confidence']:.1f}%"
                    logger.info(f"[Finalize] Late OCR vote: {best_vote['text']} ({best_vote['confidence']:.1f}%)")
                else:
                    c_data["serial_number"] = "serial_missing"
                    c_data["ocr_confidence"] = 0.0
                    logger.info(f"[Finalize] No OCR result after 5s wait, using serial_missing")

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

            # Wait another 10 seconds for recheck
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
        defect_frame = active_cycle_data.get("defect_frame_path")  # Annotated frame with defect markings
        
        # Read defects and verify holes
        with lock:
            defects = list(active_cycle_data["defects_detected"])
            if active_cycle_data["max_holes_detected"] < 3:
                defects.append("missing_holes")
            if active_cycle_data["max_rod_detected"] < 1:
                defects.append("missing_rod")
            if active_cycle_data["max_striker_detected"] < 2:
                defects.append("missing_striker")
            if active_cycle_data["max_back_hook_detected"] < 2:
                defects.append("missing_back_hook")
            if active_cycle_data["max_lock_striker_detected"] < 1:
                defects.append("missing_lock_striker")
                
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
                
        if status == "FAIL":
            folder_name = f"defect_{folder_name}"

        print(f"[Finalize Thread] Building report for Serial: {serial} in {temp_dir} with status: {status}")

        # Path of the report within the temp folder matching the folder_name
        temp_report_path = os.path.join(temp_dir, f"{folder_name}.pdf")

        # Create report PDF
        create_inspection_report(
            serial_number=serial,
            model_name="IP Cam",
            status=status,
            date_str=datetime.now().strftime("%Y-%m-%d"),
            time_str=datetime.now().strftime("%H:%M:%S"),
            traceability_id=f"TRC{folder_name}",
            front_image_path=front,
            back_image_path=back,
            ocr_serial=serial,
            confidence=conf,
            defects=defects,
            defect_image_path=defect_frame,  # Annotated front frame showing detected defects
            lock_striker_image_path=active_cycle_data.get("lock_striker_path"),
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
            # folder_name already has 'defect_' prefix if failed
            old_front_path = os.path.join(final_dir, "front.jpg")
            new_front_path = os.path.join(final_dir, f"{folder_name}_front.jpg")
            if os.path.exists(old_front_path):
                os.rename(old_front_path, new_front_path)

            old_back_path = os.path.join(final_dir, "back.jpg")
            new_back_path = os.path.join(final_dir, f"{folder_name}_back.jpg")
            if os.path.exists(old_back_path):
                os.rename(old_back_path, new_back_path)

            # Rename annotated defect frame if it exists
            old_defect_path = os.path.join(final_dir, "defect_frame.jpg")
            new_defect_path = os.path.join(final_dir, f"{folder_name}_defect.jpg")
            if os.path.exists(old_defect_path):
                os.rename(old_defect_path, new_defect_path)
                print(f"[Finalize Thread] Renamed defect frame to: {folder_name}_defect.jpg")
            
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
            
        # Hold the final result on screen for 4 seconds so the operator can read it
        time.sleep(4.0)
        
        # Automatically reset all values for the next cycle
        reset_cycle_state()

    except Exception as e:
        print(f"[Finalize Thread] Error: {e}")

def check_and_finalize_cycle(c_data):
    """Verifies if front and back have been captured, then triggers finalization."""
    if (c_data["front_path"] and 
        c_data["back_path"] and 
        not c_data["processing_thread_active"]):
        
        c_data["processing_thread_active"] = True
        threading.Thread(target=finalize_report_and_rename, args=(c_data,), daemon=True).start()

def reset_cycle_state():
    global current_cycle, active_cycle_data, cycle_count
    with lock:
        current_cycle["status"] = "Waiting for Fuel Door"
        current_cycle["cycle_number"] = f"#{cycle_count:03d}"
        current_cycle["step1_status"] = "Pending"
        current_cycle["step2_status"] = "Pending"
        current_cycle["step3_status"] = "Pending"
        current_cycle["step4_status"] = "Pending"
        current_cycle["serial"] = "------"
        current_cycle["confidence"] = "- -"
        current_cycle["result"] = "Awaiting analysis..."
        current_cycle["holes_count"] = 0
        current_cycle["rod_count"] = 0
        current_cycle["striker_count"] = 0
        current_cycle["back_hook_count"] = 0
        current_cycle["lock_striker_count"] = 0
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
            "max_rod_detected": 0,
            "max_striker_detected": 0,
            "max_back_hook_detected": 0,
            "max_lock_striker_detected": 0,
            "serial_votes": [],
            "back_frames_count": 0,
            "front_frames_count": 0,
            "serial_frames_count": 0,
            "frames_since_last_ocr_crop": 10,
            "state": "WAITING_FRONT",
            "front_box_center": None,
            "back_box_center": None,
            "front_first_seen_time": None,
            "back_first_seen_time": None,
            "remove_frames_count": 0,
            "no_panel_frames_count": 0,
            "ocr_start_time": None,
            "front_type": None,  # "standard" if 'front' detected, "circle" if 'circle_front' detected
            "defect_frame_path": None  # Path to annotated frame saved when first defect is detected on front
        }

# -------------------------------
# Background Video Processing Loop (RTSP/File Grabber & Annotator)
# -------------------------------
class CameraStreamer:
    def __init__(self):
        self.m_hDev = ctypes.c_void_p(None)
        self.m_hStream = ctypes.c_void_p(None)
        self.m_hBufferConvert = ctypes.c_void_p(None)
        self.m_isNeedConvert = ctypes.c_bool(False)
        self.latest_frame = None
        
        # Actual pixel format read back from camera after configuration
        self.actual_pixel_format = ""
        # Camera dimensions read back after configuration
        self.cam_width = 0
        self.cam_height = 0
        # Counter for one-time diagnostic logging from callback
        self._diag_frame_count = 0
        
        # Callbacks must be preserved to prevent garbage collection!
        self.EndOfFrameProc = None
        self.lock = threading.Lock()
        
        self.is_connected = False
        self.last_error_msg = ""
        
        if sdk_available:
            res = IKapC.ItkManInitialize()
            if res != IKapCDef.ITKSTATUS_OK:
                self.last_error_msg = f"Failed to initialize IKap SDK: {res}"
                print(self.last_error_msg)

    def scan_cameras(self):
        if not sdk_available:
            return []
        res, numCameras = IKapC.ItkManGetDeviceCount()
        if res != IKapCDef.ITKSTATUS_OK or numCameras == 0:
            return []
        
        cams = []
        for i in range(numCameras):
            res, devInfo = IKapC.ItkManGetDeviceInfo(i)
            if res == IKapCDef.ITKSTATUS_OK:
                cams.append({
                    "id": i,
                    "model": devInfo.FullName.decode('utf-8', errors='ignore') if devInfo.FullName else "Unknown",
                    "vendor": devInfo.VendorName.decode('utf-8', errors='ignore') if devInfo.VendorName else "I-TEK",
                    "display_name": devInfo.UserDefinedName.decode('utf-8', errors='ignore') if devInfo.UserDefinedName else f"Cam{i}"
                })
        return cams

    def connect(self, index=0, exposure=None, gain=None, gamma=None, pixel_format=None, trigger_mode=None):
        if not sdk_available:
            self.last_error_msg = "SDK unavailable."
            return False
            
        if self.is_connected or (self.m_hDev and self.m_hDev.value != 0):
            # Protect connect() against a partially opened previous connection
            self.disconnect()
            
        print("\n--- DIAGNOSTIC: I-TEK SDK DEVICE OPENING ---")
        
        # 1. Device Enumeration Result
        res, numCameras = IKapC.ItkManGetDeviceCount()
        print(f"DIAGNOSTIC: ItkManGetDeviceCount result code: {res}, Detected cameras: {numCameras}")
        print(f"DIAGNOSTIC: Attempting to open device index: {index}")
        
        # Initialize GigE specific info if applicable
        res, devInfo = IKapC.ItkManGetDeviceInfo(index)
        print(f"DIAGNOSTIC: ItkManGetDeviceInfo result code: {res}")
        if res == IKapCDef.ITKSTATUS_OK:
            dev_class = devInfo.DeviceClass.decode('utf-8', errors='ignore') if devInfo.DeviceClass else "Unknown"
            dev_name = devInfo.FullName.decode('utf-8', errors='ignore') if devInfo.FullName else "Unknown"
            dev_sn = devInfo.SerialNumber.decode('utf-8', errors='ignore') if devInfo.SerialNumber else "Unknown"
            print(f"DIAGNOSTIC: Device Info - Class: {dev_class}, Name: {dev_name}, SN: {dev_sn}")
            if devInfo.DeviceClass in [b'GigEVision', b'USB3Vision', b'GenTL']:
                gige_res, gigeInfo = IKapC.ItkManGetGigEDeviceInfo(index)
                print(f"DIAGNOSTIC: ItkManGetGigEDeviceInfo result code: {gige_res}")

        # Open device
        access_mode = IKapCDef.ITKDEV_VAL_ACCESS_MODE_EXCLUSIVE
        print(f"DIAGNOSTIC: Calling ItkDevOpen with access_mode={access_mode} (EXCLUSIVE)")
        res, self.m_hDev = IKapC.ItkDevOpen(index, access_mode)
        print(f"DIAGNOSTIC: Exact ItkDevOpen return code: {res}")
        print(f"DIAGNOSTIC: Device handle returned: {self.m_hDev}")
        
        if res != IKapCDef.ITKSTATUS_OK or not self.m_hDev or self.m_hDev.value == 0:
            self.last_error_msg = f"Failed to open device. SDK error code: {res}"
            print(self.last_error_msg)
            print("--- DIAGNOSTIC END ---\n")
            return False
            
        print("--- DIAGNOSTIC END ---\n")

        # --- Read current PixelFormat BEFORE any changes ---
        try:
            res, pf_before = IKapC.ItkDevToString(self.m_hDev, b"PixelFormat")
            pf_before_str = pf_before.decode('utf-8', errors='ignore') if isinstance(pf_before, bytes) else str(pf_before)
            print(f"PIXEL FORMAT (before config): {pf_before_str}")
        except Exception as e:
            pf_before_str = "unknown"
            print(f"Could not read PixelFormat before config: {e}")

        # --- Apply Camera Features ---
        if exposure is None:
            exposure = APP_CONFIG.get("camera", {}).get("gige", {}).get("exposure", None)
            
        if exposure is not None:
            try:
                IKapC.ItkDevSetDouble(self.m_hDev, b"ExposureTime", float(exposure))
                print(f"Applied ExposureTime: {exposure}")
            except Exception as e:
                print(f"Failed to set ExposureTime: {e}")

        if gain is None:
            gain = APP_CONFIG.get("camera", {}).get("gige", {}).get("gain", None)

        if gain is not None:
            try:
                IKapC.ItkDevSetDouble(self.m_hDev, b"Gain", float(gain))
                print(f"Applied Gain: {gain}")
            except Exception as e:
                print(f"Failed to set Gain: {e}")

        if gamma is not None:
            try:
                IKapC.ItkDevSetDouble(self.m_hDev, b"Gamma", float(gamma))
                print(f"Applied Gamma: {gamma}")
            except Exception as e:
                print(f"Failed to set Gamma: {e}")

        # --- PixelFormat configuration ---
        # UA5MEAGV-50C supports: BayerRG8, Mono8 (does NOT support BGR8/RGB8).
        # For color output, BayerRG8 is the correct native format.
        # SDK ItkBufferConvert will demosaic Bayer -> BGR888.
        if not pixel_format:
            pixel_format = APP_CONFIG.get("camera", {}).get("gige", {}).get("pixel_format", "BayerRG8")
            
        if pixel_format:
            try:
                res = IKapC.ItkDevFromString(self.m_hDev, b"PixelFormat", pixel_format.encode('utf-8'))
                if res == IKapCDef.ITKSTATUS_OK:
                    print(f"Applied PixelFormat: {pixel_format}")
                else:
                    print(f"Failed to set PixelFormat {pixel_format} (result={res}), keeping current")
            except Exception as e:
                print(f"Failed to set PixelFormat {pixel_format}: {e}")

        if trigger_mode:
            try:
                IKapC.ItkDevFromString(self.m_hDev, b"TriggerMode", trigger_mode.encode('utf-8'))
                print(f"Applied TriggerMode: {trigger_mode}")
            except Exception as e:
                print(f"Failed to set TriggerMode: {e}")

        # --- Read back ACTUAL PixelFormat after configuration ---
        try:
            res, pf_after = IKapC.ItkDevToString(self.m_hDev, b"PixelFormat")
            self.actual_pixel_format = pf_after.decode('utf-8', errors='ignore') if isinstance(pf_after, bytes) else str(pf_after)
        except Exception as e:
            self.actual_pixel_format = pf_before_str
            print(f"Could not read PixelFormat after config: {e}")

        # Read camera dimensions
        try:
            res, w_str = IKapC.ItkDevToString(self.m_hDev, b"Width")
            w_val = w_str.decode('utf-8', errors='ignore') if isinstance(w_str, bytes) else str(w_str)
            self.cam_width = int(w_val)
        except Exception:
            self.cam_width = 0
        try:
            res, h_str = IKapC.ItkDevToString(self.m_hDev, b"Height")
            h_val = h_str.decode('utf-8', errors='ignore') if isinstance(h_str, bytes) else str(h_str)
            self.cam_height = int(h_val)
        except Exception:
            self.cam_height = 0

        print(f"\n{'='*60}")
        print(f"  CAMERA CONFIGURATION SUMMARY")
        print(f"{'='*60}")
        print(f"  PIXEL FORMAT (actual): {self.actual_pixel_format}")
        print(f"  WIDTH:                 {self.cam_width}")
        print(f"  HEIGHT:                {self.cam_height}")
        print(f"{'='*60}\n")
        # -----------------------------

        # Create stream
        res, self.m_hStream = IKapC.ItkDevAllocStreamEx(self.m_hDev, 0, 5) # 5 buffers
        if res != IKapCDef.ITKSTATUS_OK:
            self.last_error_msg = "Failed to allocate stream."
            return False

        # --- Allocate conversion buffer for Bayer->BGR demosaicing ---
        # For BayerRG8 (or any Bayer format), we need the SDK to convert to BGR888.
        # Allocate at actual camera dimensions so the SDK has a properly sized destination.
        bayer_formats = ["BayerRG8", "BayerBG8", "BayerGR8", "BayerGB8",
                         "BayerRG10", "BayerBG10", "BayerGR10", "BayerGB10",
                         "BayerRG12", "BayerBG12", "BayerGR12", "BayerGB12"]
        if self.actual_pixel_format in bayer_formats and self.cam_width > 0 and self.cam_height > 0:
            res, self.m_hBufferConvert = IKapC.ItkBufferNew(
                self.cam_width, self.cam_height,
                IKapCDef.ITKBUFFER_VAL_FORMAT_BGR888
            )
            print(f"Allocated BGR888 conversion buffer ({self.cam_width}x{self.cam_height}): result={res}")
        else:
            self.m_hBufferConvert = ctypes.c_void_p(None)
            print(f"No conversion buffer needed (format={self.actual_pixel_format})")

        # Stream config
        xferMode = ctypes.c_uint32(IKapCDef.ITKSTREAM_VAL_TRANSFER_MODE_SYNCHRONOUS_WITH_PROTECT)
        startMode = ctypes.c_uint32(IKapCDef.ITKSTREAM_VAL_START_MODE_NON_BLOCK)
        IKapC.ItkStreamSetPrm(self.m_hStream, IKapCDef.ITKSTREAM_PRM_START_MODE, startMode)
        IKapC.ItkStreamSetPrm(self.m_hStream, IKapCDef.ITKSTREAM_PRM_TRANSFER_MODE, xferMode)

        # Get first buffer to check convert
        res, hBuffer = IKapC.ItkStreamGetBuffer(self.m_hStream, 0)
        res, self.m_isNeedConvert = IKapC.ItkBufferNeedAutoConvert(hBuffer)

        # Reset diagnostic frame counter
        self._diag_frame_count = 0

        # Register Callback
        self.EndOfFrameProc = ctypes.CFUNCTYPE(None, ctypes.c_void_p)(self.cbOnEndOfFrameProc)
        IKapC.ItkStreamRegisterCallback(self.m_hStream, IKapCDef.ITKSTREAM_VAL_EVENT_TYPE_END_OF_FRAME, self.EndOfFrameProc, ctypes.c_void_p(None))
        
        # Start
        res = IKapC.ItkStreamStart(self.m_hStream, 0)
        if res == IKapCDef.ITKSTATUS_OK:
            self.is_connected = True
            self.last_error_msg = ""
            print("Successfully connected to IKap GigE Camera via IKapC.")
            return True
        else:
            self.last_error_msg = "Failed to start stream."
            return False
            
    def cbOnEndOfFrameProc(self, pParam):
        res, hBuffer = IKapC.ItkStreamGetCurrentBuffer(self.m_hStream)
        if res != IKapCDef.ITKSTATUS_OK:
            return
            
        res, bufferInfo = IKapC.ItkBufferGetInfo(hBuffer)
        if bufferInfo.State != IKapCDef.ITKBUFFER_VAL_STATE_FULL and bufferInfo.State != IKapCDef.ITKBUFFER_VAL_STATE_UNCOMPLETED:
            return

        img_w = bufferInfo.ImageWidth
        img_h = bufferInfo.ImageHeight
        img_size = bufferInfo.ImageSize
        pix_fmt = self.actual_pixel_format

        # --- One-time diagnostic logging for first frame ---
        diag_first = False
        if self._diag_frame_count == 0:
            self._diag_frame_count = 1
            diag_first = True
            print(f"\n{'='*60}")
            print(f"  FIRST FRAME DIAGNOSTIC (from callback)")
            print(f"{'='*60}")
            print(f"  PIXEL FORMAT:     {pix_fmt}")
            print(f"  BUFFER ImageWidth:  {img_w}")
            print(f"  BUFFER ImageHeight: {img_h}")
            print(f"  BUFFER ImageSize:   {img_size}")
            print(f"  BUFFER TotalSize:   {bufferInfo.TotalSize}")
            print(f"  BUFFER PixelFormat: {bufferInfo.PixelFormat}")
            print(f"  BUFFER PixelDepth:  {bufferInfo.ImagePixelDepth}")

        # --- Format-aware frame conversion ---
        pix_fmt_upper = pix_fmt.upper()

        if pix_fmt_upper.startswith("BAYER"):
            # Bayer format: use SDK ItkBufferConvert to demosaic to BGR888
            if self.m_hBufferConvert and self.m_hBufferConvert.value:
                cres = IKapC.ItkBufferConvert(
                    hBuffer, self.m_hBufferConvert,
                    IKapCDef.ITKBUFFER_VAL_FORMAT_BGR888,
                    IKapCDef.ITKBUFFER_VAL_CONVERT_OPTION_AUTO_FORMAT
                )
                if cres == IKapCDef.ITKSTATUS_OK:
                    res2, convNp = IKapC.ItkBufferToNumPy(self.m_hBufferConvert)
                    if res2 == IKapCDef.ITKSTATUS_OK and convNp is not None:
                        # Reshape 1D to 3D if needed
                        if len(convNp.shape) == 1:
                            convNp = convNp.reshape((img_h, img_w, 3))
                        with self.lock:
                            self.latest_frame = convNp.copy()
                        if diag_first:
                            print(f"  SDK CONVERT:      SUCCESS (BGR888)")
                            print(f"  OUTPUT SHAPE:     {convNp.shape}")
                            print(f"  OUTPUT DTYPE:     {convNp.dtype}")
                            print(f"{'='*60}\n")
                        return
                    else:
                        if diag_first:
                            print(f"  SDK CONVERT:      NumPy extraction failed (res={res2})")
                else:
                    if diag_first:
                        print(f"  SDK CONVERT:      FAILED (result={cres})")

            # Fallback: use OpenCV Bayer demosaicing
            res3, rawNp = IKapC.ItkBufferToNumPy(hBuffer)
            if res3 == IKapCDef.ITKSTATUS_OK and rawNp is not None:
                # Reshape raw 1D to 2D (Bayer is 1 byte per pixel)
                if len(rawNp.shape) == 1:
                    if rawNp.size == img_w * img_h:
                        rawNp = rawNp.reshape((img_h, img_w))
                    else:
                        # Handle stride/padding: use stride = img_size // img_h
                        stride = img_size // img_h if img_h > 0 else img_w
                        rawNp = rawNp[:stride * img_h].reshape((img_h, stride))
                        if stride > img_w:
                            rawNp = rawNp[:, :img_w]

                # Select correct Bayer conversion code
                bayer_map = {
                    "BAYERRG8":  cv2.COLOR_BayerRG2BGR,
                    "BAYERBG8":  cv2.COLOR_BayerBG2BGR,
                    "BAYERGR8":  cv2.COLOR_BayerGR2BGR,
                    "BAYERGB8":  cv2.COLOR_BayerGB2BGR,
                    "BAYERRG10": cv2.COLOR_BayerRG2BGR,
                    "BAYERBG10": cv2.COLOR_BayerBG2BGR,
                    "BAYERGR10": cv2.COLOR_BayerGR2BGR,
                    "BAYERGB10": cv2.COLOR_BayerGB2BGR,
                    "BAYERRG12": cv2.COLOR_BayerRG2BGR,
                    "BAYERBG12": cv2.COLOR_BayerBG2BGR,
                    "BAYERGR12": cv2.COLOR_BayerGR2BGR,
                    "BAYERGB12": cv2.COLOR_BayerGB2BGR,
                }
                cvt_code = bayer_map.get(pix_fmt_upper, cv2.COLOR_BayerRG2BGR)
                bgr_frame = cv2.cvtColor(rawNp, cvt_code)

                with self.lock:
                    self.latest_frame = bgr_frame
                if diag_first:
                    print(f"  OPENCV BAYER:     Used {pix_fmt_upper} -> BGR")
                    print(f"  RAW SHAPE:        {rawNp.shape}")
                    print(f"  OUTPUT SHAPE:     {bgr_frame.shape}")
                    print(f"  OUTPUT DTYPE:     {bgr_frame.dtype}")
                    print(f"{'='*60}\n")
                    # Save diagnostic frames
                    try:
                        cv2.imwrite("diag_raw_bayer.png", rawNp)
                        cv2.imwrite("diag_bgr_result.png", bgr_frame)
                        print("  Saved diag_raw_bayer.png and diag_bgr_result.png")
                    except Exception:
                        pass
                return

        elif pix_fmt_upper == "MONO8":
            # Monochrome: get raw and convert to 3-channel BGR for pipeline compatibility
            res3, rawNp = IKapC.ItkBufferToNumPy(hBuffer)
            if res3 == IKapCDef.ITKSTATUS_OK and rawNp is not None:
                if len(rawNp.shape) == 1:
                    if rawNp.size == img_w * img_h:
                        rawNp = rawNp.reshape((img_h, img_w))
                    else:
                        stride = img_size // img_h if img_h > 0 else img_w
                        rawNp = rawNp[:stride * img_h].reshape((img_h, stride))
                        if stride > img_w:
                            rawNp = rawNp[:, :img_w]
                # Convert gray to BGR so the rest of the pipeline gets H x W x 3
                bgr_frame = cv2.cvtColor(rawNp, cv2.COLOR_GRAY2BGR)
                with self.lock:
                    self.latest_frame = bgr_frame
                if diag_first:
                    print(f"  MONO8:            gray -> BGR")
                    print(f"  RAW SHAPE:        {rawNp.shape}")
                    print(f"  OUTPUT SHAPE:     {bgr_frame.shape}")
                    print(f"{'='*60}\n")
                return

        elif pix_fmt_upper in ("BGR8", "BGR24"):
            # Already BGR: use directly
            res3, rawNp = IKapC.ItkBufferToNumPy(hBuffer)
            if res3 == IKapCDef.ITKSTATUS_OK and rawNp is not None:
                if len(rawNp.shape) == 1:
                    rawNp = rawNp.reshape((img_h, img_w, 3))
                with self.lock:
                    self.latest_frame = rawNp.copy()
                if diag_first:
                    print(f"  BGR8:             direct copy")
                    print(f"  OUTPUT SHAPE:     {rawNp.shape}")
                    print(f"{'='*60}\n")
                return

        elif pix_fmt_upper == "RGB8":
            # RGB -> BGR
            res3, rawNp = IKapC.ItkBufferToNumPy(hBuffer)
            if res3 == IKapCDef.ITKSTATUS_OK and rawNp is not None:
                if len(rawNp.shape) == 1:
                    rawNp = rawNp.reshape((img_h, img_w, 3))
                bgr_frame = cv2.cvtColor(rawNp, cv2.COLOR_RGB2BGR)
                with self.lock:
                    self.latest_frame = bgr_frame
                if diag_first:
                    print(f"  RGB8:             RGB -> BGR")
                    print(f"  OUTPUT SHAPE:     {bgr_frame.shape}")
                    print(f"{'='*60}\n")
                return

        else:
            # Unknown format: log and attempt raw extraction
            res3, rawNp = IKapC.ItkBufferToNumPy(hBuffer)
            if res3 == IKapCDef.ITKSTATUS_OK and rawNp is not None:
                if diag_first:
                    print(f"  UNKNOWN FORMAT:   '{pix_fmt}' — raw shape={rawNp.shape}, size={rawNp.size}")
                    print(f"  WARNING: Cannot determine correct conversion. Passing raw data.")
                    print(f"{'='*60}\n")
                with self.lock:
                    if len(rawNp.shape) == 1:
                        try:
                            rawNp = rawNp.reshape((img_h, img_w, 3))
                        except ValueError:
                            try:
                                rawNp = rawNp.reshape((img_h, img_w))
                                rawNp = cv2.cvtColor(rawNp, cv2.COLOR_GRAY2BGR)
                            except ValueError:
                                pass
                    self.latest_frame = rawNp.copy()

    def get_frame(self):
        with self.lock:
            if self.latest_frame is not None:
                return self.latest_frame.copy()
            return None

    def disconnect(self):
        if not sdk_available:
            return
            
        # Stop stream if running
        if self.m_hStream and self.m_hStream.value != 0:
            try:
                IKapC.ItkStreamStop(self.m_hStream)
                IKapC.ItkStreamUnregisterCallback(self.m_hStream, IKapCDef.ITKSTREAM_VAL_EVENT_TYPE_END_OF_FRAME)
                IKapC.ItkDevFreeStream(self.m_hStream)
            except Exception as e:
                print(f"Error freeing stream: {e}")
            self.m_hStream = ctypes.c_void_p(None)
            
        # Close device
        if self.m_hDev and self.m_hDev.value != 0:
            try:
                IKapC.ItkDevClose(self.m_hDev)
            except Exception as e:
                print(f"Error closing device: {e}")
            self.m_hDev = ctypes.c_void_p(None)

        # Free conversion buffer
        if self.m_hBufferConvert and self.m_hBufferConvert.value:
            try:
                IKapC.ItkBufferFree(self.m_hBufferConvert)
            except Exception as e:
                print(f"Error freeing conversion buffer: {e}")
            self.m_hBufferConvert = ctypes.c_void_p(None)
            
        # Reset connection state
        self.is_connected = False
        self.latest_frame = None
        self.actual_pixel_format = ""
        
    def update_features(self, exposure=None, gain=None, gamma=None, pixel_format=None, trigger_mode=None):
        if not self.is_connected or not self.m_hDev or self.m_hDev.value == 0:
            return False, "Camera is not connected."
            
        with self.lock:
            # --- Apply Camera Features ---
            if exposure is not None:
                try:
                    IKapC.ItkDevSetDouble(self.m_hDev, b"ExposureTime", float(exposure))
                    print(f"Updated ExposureTime: {exposure}")
                except Exception as e:
                    print(f"Failed to set ExposureTime: {e}")

            if gain is not None:
                try:
                    IKapC.ItkDevSetDouble(self.m_hDev, b"Gain", float(gain))
                    print(f"Updated Gain: {gain}")
                except Exception as e:
                    print(f"Failed to set Gain: {e}")

            if gamma is not None:
                try:
                    IKapC.ItkDevSetDouble(self.m_hDev, b"Gamma", float(gamma))
                    print(f"Updated Gamma: {gamma}")
                except Exception as e:
                    print(f"Failed to set Gamma: {e}")

            if pixel_format:
                try:
                    IKapC.ItkDevFromString(self.m_hDev, b"PixelFormat", pixel_format.encode('utf-8'))
                    print(f"Updated PixelFormat: {pixel_format}")
                except Exception as e:
                    print(f"Failed to set PixelFormat: {e}")

            if trigger_mode:
                try:
                    IKapC.ItkDevFromString(self.m_hDev, b"TriggerMode", trigger_mode.encode('utf-8'))
                    print(f"Updated TriggerMode: {trigger_mode}")
                except Exception as e:
                    print(f"Failed to set TriggerMode: {e}")
            return True, "Features updated successfully."

cam = CameraStreamer()
atexit.register(cam.disconnect)

def video_processing_loop():
    global video_cap, is_processing, stream_source, latest_raw_frame, latest_annotated_frame, current_detections
    consecutive_failures = 0
    _vloop_counter = 0
    print("[VideoLoop] Thread started!")

    while True:
        try:
            _vloop_counter += 1
            if _vloop_counter <= 3 or _vloop_counter % 100 == 0:
                print(f"[VideoLoop] tick #{_vloop_counter}, is_processing={is_processing}, stream_source={stream_source!r}", flush=True)

            if stream_source == "gige":
                if not cam.is_connected:
                    time.sleep(0.05)
                    continue
                frame = cam.get_frame()
                if frame is None:
                    time.sleep(0.01)
                    continue
                ret = True
            else:
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

            # Resize frame to standard 480px width first to speed up JPEG encoding and keep annotations crisp
            h_ann, w_ann = frame.shape[:2]
            UI_WIDTH = 480 # Reduced resolution for UI to prevent getting stuck
            if w_ann > UI_WIDTH:
                scale_ann = UI_WIDTH / w_ann
                annotated_frame = cv2.resize(frame, (UI_WIDTH, int(h_ann * scale_ann)))
            else:
                scale_ann = 1.0
                annotated_frame = frame.copy()

            with lock:
                dets = list(current_detections)

            # First pass: collect and draw all masks on a single overlay
            overlay = None
            for det in dets:
                class_name = det["class_name"]
                if class_name in ["line_mark", "dent", "bulge", "damage"]:
                    mask_pts = det.get("mask")
                    if mask_pts is not None:
                        color = (0, 0, 255) if class_name in ["dent", "bulge", "damage"] else (0, 255, 255)
                        pts = np.array(mask_pts, np.int32)
                        pts[:, 0] = (pts[:, 0] * scale_ann).astype(int)
                        pts[:, 1] = (pts[:, 1] * scale_ann).astype(int)
                        pts = pts.reshape((-1, 1, 2))
                        
                        if overlay is None:
                            overlay = annotated_frame.copy()
                            
                        cv2.fillPoly(overlay, [pts], color)

            # Apply all semi-transparent masks at once
            if overlay is not None:
                cv2.addWeighted(overlay, 0.3, annotated_frame, 0.7, 0, annotated_frame)

            # Second pass: draw bounding boxes, borders, and labels
            for det in dets:
                x1 = int(det["box"][0] * scale_ann)
                y1 = int(det["box"][1] * scale_ann)
                x2 = int(det["box"][2] * scale_ann)
                y2 = int(det["box"][3] * scale_ann)

                class_name = det["class_name"]
                
                color = (0, 255, 0)
                if class_name in ["front", "circle_front"]:
                    color = (0, 165, 255)
                elif class_name in ["back", "circle_back", "cricle_back"]:
                    color = (255, 0, 255)
                elif class_name in ["serial", "serial_area"]:
                    color = (255, 255, 0)
                elif class_name in ["dent", "bulge", "damage"]:
                    color = (0, 0, 255)
                elif class_name == "line_mark":
                    color = (0, 255, 255)

                if class_name in ["line_mark", "dent", "bulge", "damage"] and det.get("mask") is not None:
                    pts = np.array(det["mask"], np.int32)
                    pts[:, 0] = (pts[:, 0] * scale_ann).astype(int)
                    pts[:, 1] = (pts[:, 1] * scale_ann).astype(int)
                    pts = pts.reshape((-1, 1, 2))
                    cv2.polylines(annotated_frame, [pts], True, color, 2)
                else:
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)

                # Draw label without confidence percentage
                label = f"{class_name}"
                (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                text_x = x1
                text_y = max(y1 - 5, text_h + 5)
                cv2.rectangle(annotated_frame, (text_x, text_y - text_h - 4), (text_x + text_w, text_y + 2), color, -1)
                cv2.putText(annotated_frame, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

            # Encode annotated frame to JPEG with lower quality for UI performance
            ret2, buffer = cv2.imencode('.jpg', annotated_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 40])
            if ret2:
                with lock:
                    latest_annotated_frame = buffer.tobytes()

            # Speed throttle - read at true video FPS
            if isinstance(stream_source, str) and not stream_source.startswith("rtsp"):
                fps = video_cap.get(cv2.CAP_PROP_FPS) if video_cap else 30
                if fps <= 0:
                    fps = 30
                
                target_frame_time = 1.0 / fps
                
                # Use a global dictionary to store the last time so we don't attach attributes to cv2 object
                if not hasattr(video_cap, 'isOpened'):
                    pass
                
                if 'last_video_time' not in globals():
                    global last_video_time
                    last_video_time = time.time()
                
                elapsed = time.time() - last_video_time
                
                # If we are falling behind (processing took too long), skip frames to catch up and keep it smooth
                while elapsed > target_frame_time:
                    video_cap.grab()  # discard a frame instantly
                    elapsed -= target_frame_time
                
                sleep_time = max(0.001, target_frame_time - elapsed)
                time.sleep(sleep_time)
                
                last_video_time = time.time()
            else:
                time.sleep(0.001)

        except Exception as e:
            import traceback
            print(f"[VideoLoop] EXCEPTION: {e}")
            traceback.print_exc()
            time.sleep(0.5)


# -------------------------------
# Dedicated Background YOLO Worker Thread
# -------------------------------
def yolo_worker_loop():
    global YOLO_MODEL, latest_raw_frame, current_detections, active_cycle_data, current_cycle
    init_models()
    
    _debug_counter = 0
    while True:
        if not is_processing or latest_raw_frame is None:
            time.sleep(0.01)
            continue
            
        # Grab copy of latest raw frame
        with lock:
            frame_to_process = latest_raw_frame.copy()
            state = active_cycle_data.get("state", "WAITING_FRONT")
        
        _debug_counter += 1
        if _debug_counter % 30 == 0:
            print(f"[YOLO] Running inference frame #{_debug_counter}, state={state}")
            
        h_orig, w_orig = frame_to_process.shape[:2]
        
        # Resize frame for faster YOLO inference (user trained at 432x432, 448 is closest stride-32 multiple)
        scale = 1.0
        if w_orig > 448:
            scale = 448 / w_orig
            frame_resized = cv2.resize(frame_to_process, (448, int(h_orig * scale)))
        else:
            frame_resized = frame_to_process.copy()
            
        try:
            # Determine minimum threshold for YOLO to return all relevant boxes
            min_thresh = min(CLASS_CONF_THRESHOLDS.values()) if CLASS_CONF_THRESHOLDS else YOLO_CONF_THRESHOLD
            run_thresh = min(float(min_thresh), float(YOLO_CONF_THRESHOLD))
            
            results = YOLO_MODEL(frame_resized, verbose=False, conf=run_thresh, task='segment', imgsz=448)
            new_detections = []
            frame_holes = 0
            frame_rod = 0
            frame_striker = 0
            frame_back_hook = 0
            frame_lock_striker = 0
            frame_defects = []
            
            # Frame with annotations for saving
            annotated_frame = frame_to_process.copy()
            has_front_detected = False
            has_back_detected = False
            front_box = None
            back_box = None
            front_class = None  # Will be 'front' or 'circle_front'
            back_class = None   # Will be 'back', 'circle_back', or 'cricle_back'
            
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
                    class_name = names[cls_id].lower()
                    conf = float(box.conf[0].cpu().item())
                    
                    if conf < CLASS_CONF_THRESHOLDS.get(class_name, YOLO_CONF_THRESHOLD):
                        continue
                        
                    if class_name in ["front", "circle_front"]:
                        has_front_detected = True
                    elif class_name in ["back", "circle_back", "cricle_back", "serial", "serial_area"]:
                        has_back_detected = True
                
                for box_idx, box in enumerate(boxes):
                    xyxy_resized = box.xyxy[0].cpu().numpy().astype(int)
                    cls_id = int(box.cls[0].cpu().item())
                    class_name = names[cls_id].lower()
                    
                    # We process all classes
                    conf = float(box.conf[0].cpu().item())
                    
                    if conf < CLASS_CONF_THRESHOLDS.get(class_name, YOLO_CONF_THRESHOLD):
                        continue
                        
                    # Removed the condition that ignores defects if front is not detected
                    # so that defects are always shown if found.                    
                    mask_polygon = None
                    if getattr(results[0], 'masks', None) is not None and len(results[0].masks.xy) > box_idx:
                        segment = results[0].masks.xy[box_idx]
                        if segment is not None and len(segment) > 0:
                            segment_scaled = segment.copy()
                            segment_scaled[:, 0] = segment_scaled[:, 0] / scale
                            segment_scaled[:, 1] = segment_scaled[:, 1] / scale
                            mask_polygon = segment_scaled.astype(int).tolist()
                    
                    # Map coordinates back to original frame size
                    x1 = int(xyxy_resized[0] / scale)
                    y1 = int(xyxy_resized[1] / scale)
                    x2 = int(xyxy_resized[2] / scale)
                    y2 = int(xyxy_resized[3] / scale)
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w_orig, x2), min(h_orig, y2)
                    
                    # --- NEW LOGIC: Empty Tray Filter ---
                    # Apply empty tray filtering to all classes
                    # Load config values
                    presence_cfg = APP_CONFIG.get("ai", {}).get("part_presence", {})
                    blue_t = presence_cfg.get("blue_thresh", 0.75)
                    metal_t = presence_cfg.get("metal_min_ratio", 0.15)
                    lap_t = presence_cfg.get("laplacian_variance_min", 150.0)
                    
                    if class_name not in ["dent", "bulge", "line_mark", "damage"]:
                        roi = frame_to_process[y1:y2, x1:x2]
                        if not has_part(roi, blue_t, metal_t, lap_t):
                            logger.info(f"Filtered out empty tray misclassified as '{class_name}' (conf: {conf:.2f})")
                            continue # Skip this bounding box
                    else:
                        # Defect classes are only valid on the FRONT side
                        if not has_front_detected:
                            continue # Skip this defect bounding box
                        
                        # Fast and accurate filter: remove tiny false positive defects based on area
                        box_area = (x2 - x1) * (y2 - y1)
                        min_defect_area = APP_CONFIG.get("ai", {}).get("min_defect_area", 200)
                        if box_area < min_defect_area:
                            logger.info(f"Filtered out {class_name} due to small area ({box_area} < {min_defect_area})")
                            continue
                    # ------------------------------------

                    new_detections.append({
                        "box": [x1, y1, x2, y2],
                        "class_name": class_name, "mask": mask_polygon,
                        "conf": conf
                    })
                    
                    # Draw annotations for saved images
                    color = (0, 255, 0)
                    if class_name in ["front", "circle_front"]: color = (0, 165, 255)
                    elif class_name in ["back", "circle_back", "cricle_back"]: color = (255, 0, 255)
                    elif class_name in ["serial", "serial_area"]: color = (255, 255, 0)
                    elif class_name in ["dent", "bulge", "damage"]: color = (0, 0, 255)
                    elif class_name == "line_mark": color = (0, 255, 255)
                    
                    if class_name in ["line_mark", "dent", "bulge", "damage"]:
                        if mask_polygon is not None:
                            pts = np.array(mask_polygon, np.int32).reshape((-1, 1, 2))
                            overlay = annotated_frame.copy()
                            cv2.fillPoly(overlay, [pts], color)
                            cv2.addWeighted(overlay, 0.3, annotated_frame, 0.7, 0, annotated_frame)
                            cv2.polylines(annotated_frame, [pts], True, color, 3)
                        else:
                            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 3)
                    else:
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
                    
                    if class_name in ["dent", "bulge", "line_mark", "damage"]:
                        frame_defects.append(class_name)
                        
                    if class_name == "hole" or class_name == "holes":
                        if has_back_detected or (not has_front_detected and not has_back_detected):
                            frame_holes += 1
                    elif class_name == "rod":
                        frame_rod += 1
                    elif class_name == "striker":
                        frame_striker += 1
                    elif class_name == "back_hook":
                        frame_back_hook += 1
                    elif class_name == "lock_striker":
                        frame_lock_striker += 1
                    elif class_name in ["front", "circle_front"]:
                        front_box = (x1, y1, x2, y2)
                        front_class = class_name  # Track whether it's 'front' or 'circle_front'
                    elif class_name in ["back", "circle_back", "cricle_back"]:
                        back_box = (x1, y1, x2, y2)
                        back_class = class_name  # Track the actual detected back class

                        
            # --- Defect Frame Save ---
            # When defects are detected on the front panel, save the annotated frame once per cycle.
            # annotated_frame already has all defect masks and bounding boxes drawn on it.
            # This gives us a 3rd image (in addition to clean front + clean back) for the report.
            if frame_defects and has_front_detected and active_cycle_data["temp_folder"] is not None:
                with lock:
                    defect_frame_already_saved = active_cycle_data.get("defect_frame_path") is not None
                if not defect_frame_already_saved:
                    defect_file = os.path.join(active_cycle_data["temp_folder"], "defect_frame.jpg")
                    cv2.imwrite(defect_file, annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    with lock:
                        active_cycle_data["defect_frame_path"] = defect_file
                    logger.info(f"[Defect Frame] Saved annotated defect frame: {defect_file} (defects: {frame_defects})")

            # --- Lock Striker Frame Save ---
            if frame_lock_striker > 0 and active_cycle_data["temp_folder"] is not None:
                with lock:
                    ls_frame_already_saved = active_cycle_data.get("lock_striker_path") is not None
                if not ls_frame_already_saved:
                    ls_file = os.path.join(active_cycle_data["temp_folder"], "lock_striker.jpg")
                    # Save annotated_frame to show bounding box, or frame_to_process for clean image
                    cv2.imwrite(ls_file, annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    with lock:
                        active_cycle_data["lock_striker_path"] = ls_file
                    logger.info(f"[Lock Striker] Saved annotated lock striker frame: {ls_file}")

            # OCR logic (runs asynchronously)
            # GUARD: Only attempt OCR after front has been captured (temp_folder exists)
            # and we are in WAITING_BACK state. This prevents stale OCR threads from
            # contaminating new cycles.
            ocr_state = active_cycle_data["state"]
            if active_cycle_data["serial_number"] is None and active_cycle_data["temp_folder"] is not None and ocr_state == "WAITING_BACK":
                # Find the best target for OCR according to priority
                ocr_target_det = None
                
                # Only trigger OCR when back panel (or circle_back) is confirmed in the frame
                back_in_frame = any(d["class_name"] in ["back", "circle_back", "cricle_back"] for d in new_detections)

                # Check for classes in priority order - only proceed if back is visible
                if back_in_frame:
                    for cls_target in ["serial", "serial_area"]:
                        matches = [d for d in new_detections if d["class_name"] == cls_target and d["conf"] >= OCR_CONF_THRESHOLD]
                        if matches:
                            ocr_target_det = max(matches, key=lambda x: x["conf"])
                            break
                        
                if ocr_target_det:
                    with lock:
                        if active_cycle_data["ocr_start_time"] is None:
                            active_cycle_data["ocr_start_time"] = time.time()
                        time_elapsed = time.time() - active_cycle_data["ocr_start_time"]

                    class_name = ocr_target_det["class_name"]
                    is_ocr_target = False

                    if class_name in ["serial", "serial_area"]:
                        # Give the part 0.4 seconds to settle before taking the OCR crop to avoid motion blur on the long-exposure camera
                        if time_elapsed >= 0.4:
                            is_ocr_target = True
                    else:
                        is_ocr_target = False

                    if is_ocr_target:
                        with lock:
                            should_start_ocr = not active_cycle_data["ocr_thread_active"]

                        if should_start_ocr:
                            with lock:
                                active_cycle_data["ocr_thread_active"] = True

                            print(f"\n[OCR TARGET]")
                            print(f"class = {class_name}")
                            print(f"confidence = {ocr_target_det['conf']}")
                            print(f"bounding_box = {ocr_target_det['box']}")
                            print(f"source_type = {class_name}\n")

                            x1, y1, x2, y2 = ocr_target_det["box"]

                            # If we're cropping from serial_area, prefer the tighter 'serial' box inside it
                            if class_name == "serial_area":
                                serial_matches = [d for d in new_detections if d["class_name"] == "serial"]
                                if serial_matches:
                                    best_serial = max(serial_matches, key=lambda x: x["conf"])
                                    x1, y1, x2, y2 = best_serial["box"]
                                    print(f"[OCR] Using tighter 'serial' box inside serial_area: {best_serial['box']}")
                            
                            # Extract the expected serial-number region from this back-side crop
                            # If it's the whole back, we still use the bounding box as the source region
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

                            print(f"[OCR CROP]")
                            print(f"crop shape = {crop.shape}")
                            print(f"crop path = RAM")
                            print(f"source class = {class_name}\n")

                            print(f"[OCR REQUEST]")
                            print(f"sending crop to OCR server\n")

                            next_crop_num = len(active_cycle_data["serial_votes"]) + 1
                            threading.Thread(
                                target=process_ocr_async,
                                args=(crop, active_cycle_data["temp_folder"], next_crop_num, active_cycle_data),
                                daemon=True
                            ).start()
                        elif class_name in ["circle_back", "cricle_back", "back"]:
                            # If we couldn't start OCR (thread busy), it's not a complete failure yet, just skipping this frame
                            pass
                    else:
                        if class_name in ["circle_back", "cricle_back", "back"]:
                            print(f"[OCR ERROR] Back-side detection found but OCR crop was not generated")
                else:
                    with lock:
                        t_start = active_cycle_data["ocr_start_time"]
                    if t_start is not None and (time.time() - t_start) > 3.5:
                        print(f"[OCR ERROR] OCR was never triggered")

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
                            active_cycle_data["front_first_seen_time"] = None
                            
                        active_cycle_data["front_box_center"] = (cx, cy)
                        active_cycle_data["front_frames_count"] += 1
                        
                        if active_cycle_data.get("front_first_seen_time") is None:
                            active_cycle_data["front_first_seen_time"] = time.time()
                            
                        time_stable = time.time() - active_cycle_data["front_first_seen_time"]
                        
                        active_cycle_data["front_missing_frames"] = 0
                        
                        if time_stable >= 0.2:
                            # Save full raw feed image without annotations and without cropping
                            full_image = frame_to_process.copy()
                            front_file = os.path.join(active_cycle_data["temp_folder"], "front.jpg")
                            cv2.imwrite(front_file, full_image, [cv2.IMWRITE_JPEG_QUALITY, 95])
                            
                            active_cycle_data["front_path"] = front_file
                            # Record which front panel type was seen so back-panel mismatch can be detected
                            active_cycle_data["front_type"] = "circle" if front_class == "circle_front" else "standard"
                            logger.info(f"[State] Front captured as type='{active_cycle_data['front_type']}' (class='{front_class}')")
                            current_cycle["step2_status"] = "OK"
                            active_cycle_data["state"] = "WAITING_LOCK_STRIKER"
                            for d in frame_defects:
                                active_cycle_data["defects_detected"].add(d)

                    else:
                        active_cycle_data["front_missing_frames"] = active_cycle_data.get("front_missing_frames", 0) + 1
                        if active_cycle_data["front_missing_frames"] > 10:
                            active_cycle_data["front_frames_count"] = 0
                            active_cycle_data["front_first_seen_time"] = None
                            
                elif state == "WAITING_LOCK_STRIKER":
                    current_cycle["status"] = "Waiting for Lock Striker"
                    current_cycle["instruction"] = "SHOW LOCK STRIKER"
                    current_cycle["instruction_color"] = "blue"
                    
                    is_back_visible = (back_box is not None) or has_serial_detected
                    
                    if is_back_visible:
                        current_cycle["status"] = "Skipped Lock Striker!"
                        current_cycle["instruction"] = "SHOW LOCK STRIKER FIRST"
                        current_cycle["instruction_color"] = "red"
                    
                    # Check if the lock striker was captured
                    with lock:
                        ls_saved = active_cycle_data.get("lock_striker_path") is not None
                    if ls_saved:
                        current_cycle["step3_status"] = "OK"
                        active_cycle_data["state"] = "WAITING_BACK"
                        
                elif state == "WAITING_BACK":
                    current_cycle["status"] = "Waiting for Back Panel"
                    current_cycle["instruction"] = "FLIP TO BACK SIDE"
                    current_cycle["instruction_color"] = "blue"
                    
                    is_back_visible = (back_box is not None) or has_serial_detected
                    
                    if is_back_visible and not has_front_detected:
                        # --- Panel Type Mismatch Check ---
                        # Only run when the back panel itself (not just serial/holes sub-features) is detected
                        front_type = active_cycle_data.get("front_type")
                        is_mismatched = False
                        if back_box is not None and front_type is not None:
                            if front_type == "standard" and back_class in ["circle_back", "cricle_back"]:
                                is_mismatched = True
                            elif front_type == "circle" and back_class == "back":
                                is_mismatched = True

                        if is_mismatched:
                            # Wrong part type placed — alert operator and block capture
                            expected = "STANDARD BACK" if front_type == "standard" else "CIRCLE BACK"
                            logger.warning(f"[Mismatch] Front type='{front_type}' but detected back_class='{back_class}'. Blocking capture.")
                            current_cycle["status"] = "Part Type Mismatch!"
                            current_cycle["instruction"] = "PLACE CORRECT PART"
                            current_cycle["instruction_color"] = "red"
                            # Reset back-stability counters so capture does not proceed
                            active_cycle_data["back_frames_count"] = 0
                            active_cycle_data["back_first_seen_time"] = None
                        else:
                            # Correct panel (or only sub-features visible) — proceed normally
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
                                active_cycle_data["back_first_seen_time"] = None
                                
                            active_cycle_data["back_box_center"] = (cx, cy)
                            active_cycle_data["back_frames_count"] += 1
                            
                            if active_cycle_data.get("back_first_seen_time") is None:
                                active_cycle_data["back_first_seen_time"] = time.time()
                                
                            time_stable = time.time() - active_cycle_data["back_first_seen_time"]
                            
                            ocr_done = active_cycle_data.get("serial_number") is not None
                            ocr_start = active_cycle_data.get("ocr_start_time")
                            ocr_timeout = (ocr_start is not None) and (time.time() - ocr_start > 5.0)
                            
                            if time_stable >= 0.0 or ocr_done:
                                # Delay the transition to finalization until the OCR thread has actually successfully completed or timed out
                                if ocr_done or ocr_timeout:
                                    # Save full annotated image without cropping
                                    full_raw = frame_to_process.copy()
                                    back_file = os.path.join(active_cycle_data["temp_folder"], "back.jpg")
                                    cv2.imwrite(back_file, full_raw, [cv2.IMWRITE_JPEG_QUALITY, 95])
                                    active_cycle_data["back_path"] = back_file
                                    
                                    current_cycle["step4_status"] = "OK"
                                    for d in frame_defects:
                                        active_cycle_data["defects_detected"].add(d)
                                    
                                    # Go straight to finalization
                                    active_cycle_data["state"] = "WAITING_REMOVE"
                                    check_and_finalize_cycle(active_cycle_data)
                                else:
                                    current_cycle["instruction"] = "DETECTING SERIAL NUMBER..."
                                    active_cycle_data["back_frames_count"] = 1 # Keep it below threshold until serial is seen
                    else:
                        active_cycle_data["back_frames_count"] = 0
                        active_cycle_data["back_first_seen_time"] = None

                        
                elif state == "WAITING_REMOVE":
                    # Check for ANY detection (panel or sub-features from circular shapes)
                    any_lid_visible = has_front_detected or has_back_detected or len(sub_feature_boxes) > 0
                    if not any_lid_visible:
                        active_cycle_data["remove_frames_count"] += 1
                        # Immediately clear the serial from the UI on the very first frame
                        # the part is gone, so the client sees the screen is ready for next part
                        if active_cycle_data["remove_frames_count"] == 1:
                            current_cycle["serial"] = "------"
                            current_cycle["confidence"] = "- -"
                            current_cycle["status"] = "Part removed. Saving report..."
                            current_cycle["instruction"] = "REMOVE COMPLETE - SAVING..."
                            current_cycle["instruction_color"] = "orange"
                        if active_cycle_data["remove_frames_count"] >= 3:
                            needs_reset = True
                    elif has_front_detected and not has_back_detected:
                        # A NEW front panel appeared — the operator has placed the next part.
                        # Reset immediately so the new cycle starts cleanly without the old serial.
                        logger.info("[State] New FRONT detected in WAITING_REMOVE — auto-resetting for next cycle.")
                        needs_reset = True
                    else:
                        active_cycle_data["remove_frames_count"] = 0
                        
                # Update UI data outside state transitions
                if state in ["WAITING_FRONT", "WAITING_BACK", "WAITING_REMOVE"]:
                    current_cycle["holes_count"] = min(3, max(current_cycle["holes_count"], frame_holes))
                    current_cycle["rod_count"] = min(1, max(current_cycle["rod_count"], frame_rod))
                    current_cycle["striker_count"] = min(2, max(current_cycle["striker_count"], frame_striker))
                    current_cycle["back_hook_count"] = min(2, max(current_cycle["back_hook_count"], frame_back_hook))
                    current_cycle["lock_striker_count"] = min(1, max(current_cycle["lock_striker_count"], frame_lock_striker))
                    
                    # Continuously accumulate defects from every frame (not just at capture time)
                    for d in frame_defects:
                        active_cycle_data["defects_detected"].add(d)
                    current_cycle["defects"] = list(active_cycle_data["defects_detected"])
                    if active_cycle_data["temp_folder"] is not None:
                        active_cycle_data["max_holes_detected"] = min(3, max(active_cycle_data["max_holes_detected"], frame_holes))
                        active_cycle_data["max_rod_detected"] = min(1, max(active_cycle_data["max_rod_detected"], frame_rod))
                        active_cycle_data["max_striker_detected"] = min(2, max(active_cycle_data["max_striker_detected"], frame_striker))
                        active_cycle_data["max_back_hook_detected"] = min(2, max(active_cycle_data["max_back_hook_detected"], frame_back_hook))
                        active_cycle_data["max_lock_striker_detected"] = min(1, max(active_cycle_data["max_lock_striker_detected"], frame_lock_striker))
            if needs_reset:
                reset_cycle_state()
                
        except Exception as e:
            import traceback
            logger.error(f"[YOLO Thread Inference Error] {e}")
            logger.error(traceback.format_exc())
            if 'frame_to_process' in locals() and frame_to_process is not None:
                logger.error(f"frame_to_process shape: {frame_to_process.shape}")
            
        time.sleep(0.01)

# -------------------------------
# Disk Space Management (Auto Cleanup)
# -------------------------------
def auto_cleanup_loop():
    """Runs continuously in the background, deleting old files once per day."""
    while True:
        try:
            now = time.time()
            data_dir = "lid_data"
            if os.path.exists(data_dir):
                for folder in os.listdir(data_dir):
                    folder_path = os.path.join(data_dir, folder)
                    # We look for date-formatted folders like "2026-08-22"
                    if os.path.isdir(folder_path) and re.match(r"\d{4}-\d{2}-\d{2}", folder):
                        folder_mtime = os.path.getmtime(folder_path)
                        age_days = (now - folder_mtime) / (24 * 3600)
                        if age_days > RETENTION_DAYS:
                            shutil.rmtree(folder_path, ignore_errors=True)
                            print(f"[Cleanup] Deleted old data folder: {folder}")
        except Exception as e:
            print(f"[Cleanup Error] {e}")
        
        # Sleep for 24 hours
        time.sleep(24 * 3600)

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

@app.route('/scan_cameras', methods=['GET'])
def scan_cameras():
    global cam
    return jsonify({'cameras': cam.scan_cameras()})

@app.route('/connect_gige', methods=['POST'])
def connect_gige():
    global cam
    if cam.is_connected:
        return jsonify({"status": "success", "message": "GigE Camera already connected."})
        
    data = request.json or {}
    exposure = data.get('exposure')
    gain = data.get('gain')
    gamma = data.get('gamma')
    pixel_format = data.get('pixel_format')
    trigger_mode = data.get('trigger_mode')
    
    if cam.connect(0, exposure=exposure, gain=gain, gamma=gamma, pixel_format=pixel_format, trigger_mode=trigger_mode):
        global stream_source, is_processing, video_cap
        if video_cap:
            video_cap.release()
            video_cap = None
        stream_source = "gige"
        is_processing = True
        reset_cycle_state()
        return jsonify({"status": "success", "message": "GigE Camera connected successfully!"})
    else:
        return jsonify({"status": "error", "message": cam.last_error_msg})

@app.route('/disconnect_gige', methods=['POST'])
def disconnect_gige():
    global cam, is_processing, stream_source
    try:
        cam.disconnect()
        is_processing = False
        stream_source = 0
        return jsonify({"status": "success", "message": "Camera disconnected successfully."})
    except Exception as e:
        return jsonify({"status": "error", "message": f"Failed to disconnect: {str(e)}"})

@app.route('/update_gige_features', methods=['POST'])
def update_gige_features():
    global cam
    data = request.json or {}
    exposure = data.get('exposure')
    gain = data.get('gain')
    gamma = data.get('gamma')
    pixel_format = data.get('pixel_format')
    trigger_mode = data.get('trigger_mode')
    
    success, msg = cam.update_features(exposure, gain, gamma, pixel_format, trigger_mode)
    if success:
        return jsonify({"status": "success", "message": msg})
    else:
        return jsonify({"status": "error", "message": msg})

@app.route('/camera_status', methods=['GET'])
def camera_status():
    global cam
    return jsonify({
        "connected": cam.is_connected,
        "status": "Connected" if cam.is_connected else "Disconnected",
        "fps": 30,
        "resolution": "High-Res",
        "model": "IKap GigE Camera" if cam.is_connected else ""
    })

@app.route('/camera_diag', methods=['GET'])
def camera_diag():
    """Diagnostic endpoint to report actual camera pixel format and frame info."""
    global cam
    diag = {
        "connected": cam.is_connected,
        "actual_pixel_format": cam.actual_pixel_format,
        "cam_width": cam.cam_width,
        "cam_height": cam.cam_height,
    }
    frame = cam.get_frame()
    if frame is not None:
        diag["frame_shape"] = list(frame.shape)
        diag["frame_dtype"] = str(frame.dtype)
        diag["frame_channels"] = frame.shape[2] if len(frame.shape) == 3 else 1
    else:
        diag["frame_shape"] = None
        diag["frame_dtype"] = None
        diag["frame_channels"] = None
    return jsonify(diag)
@app.route('/status')
def status():
    global is_processing
    resp = dict(current_cycle)
    resp["is_processing"] = is_processing
    resp["capture_state"] = active_cycle_data.get("state", "WAITING_FRONT")
    return jsonify(resp)

if __name__ == '__main__':
    # Initialize workspace folders
    os.makedirs("lid_data", exist_ok=True)
    init_models()
    
    import subprocess
    import atexit
    
    logger.info("[System] Automatically starting PaddleOCR Server...")
    try:
        ocr_log = open("logs/ocr_server.log", "w", buffering=1)
        # Try specific Python 3.11 launcher first (this runs the server with global GPU python)
        try:
            py311_path = subprocess.check_output(["py", "-3.11", "-c", "import sys; print(sys.executable)"]).decode().strip()
            ocr_process = subprocess.Popen([py311_path, "paddleocr_server.py"], stdout=ocr_log, stderr=subprocess.STDOUT, env=dict(os.environ, PYTHONUNBUFFERED="1"))
        except Exception:
            ocr_process = subprocess.Popen([sys.executable, "paddleocr_server.py"], stdout=ocr_log, stderr=subprocess.STDOUT, env=dict(os.environ, PYTHONUNBUFFERED="1"))
            
        def cleanup_ocr():
            logger.info("[System] Shutting down PaddleOCR Server...")
            try:
                ocr_process.terminate()
                ocr_process.wait(timeout=2)
                ocr_log.close()
            except Exception:
                try:
                    ocr_process.kill()
                except:
                    pass
                
        atexit.register(cleanup_ocr)
    except Exception as e:
        logger.error(f"[System] Failed to start PaddleOCR server: {e}")
    
    # Start the video processing background thread
    processing_thread = threading.Thread(target=video_processing_loop, daemon=True)
    processing_thread.start()
    
    # Start the YOLO inference worker thread
    yolo_thread = threading.Thread(target=yolo_worker_loop, daemon=True)
    yolo_thread.start()
    
    # Start the auto-cleanup thread
    cleanup_thread = threading.Thread(target=auto_cleanup_loop, daemon=True)
    cleanup_thread.start()
    
    app.run(host=APP_CONFIG.get("web", {}).get("host", "0.0.0.0"), 
            port=APP_CONFIG.get("web", {}).get("port", 5000), 
            debug=False, threaded=True)

