"""
PaddleOCR Microservice - runs on Python 3.12
Exposes a simple HTTP endpoint so the main app.py (Python 3.14) can use PaddleOCR.

Usage: py -3.12 paddleocr_server.py
Endpoint: POST http://127.0.0.1:5001/ocr  (body: multipart image OR raw bytes)
"""

import base64
import json
import logging
import traceback
from http.server import BaseHTTPRequestHandler, HTTPServer

import cv2
import numpy as np

import sys
logging.basicConfig(level=logging.INFO, format='%(asctime)s [OCR-Server] %(message)s', stream=sys.stdout)

# Custom handler to force flush
class FlushFileHandler(logging.StreamHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()

logging.getLogger().handlers = [FlushFileHandler(sys.stdout)]

logging.info("Initializing PaddleOCR engine...")
from paddleocr import PaddleOCR
import paddle

ocr = None
try:
    ocr = PaddleOCR(lang='en')
    logging.info("PaddleOCR successfully initialized.")
except Exception as e:
    logging.error(f"PaddleOCR initialization failed: {e}")
    try:
        ocr = PaddleOCR()
        logging.info("PaddleOCR initialized with default config.")
    except Exception as ex:
        logging.error(f"Fallback PaddleOCR initialization failed: {ex}")


def run_ocr(image):
    """Run PaddleOCR with fast recognition fallback for cropped serial regions."""
    def parse_res(res_list):
        parsed = []
        if not res_list or not isinstance(res_list, list):
            return parsed
        for res in res_list:
            if not res:
                continue
            if hasattr(res, 'keys') and 'rec_texts' in res:
                texts = res.get('rec_texts', [])
                scores = res.get('rec_scores', [])
                for t, s in zip(texts, scores):
                    parsed.append({"rec_text": t, "rec_score": float(s)})
            elif isinstance(res, list):
                for line in res:
                    if isinstance(line, list) and len(line) == 2 and isinstance(line[1], tuple):
                        parsed.append({
                            "rec_text": line[1][0],
                            "rec_score": float(line[1][1])
                        })
                    elif isinstance(line, tuple) and len(line) == 2:
                        parsed.append({
                            "rec_text": line[0],
                            "rec_score": float(line[1])
                        })
        return parsed

    # 1. Try standard detection + recognition
    try:
        raw_result = ocr.ocr(image, det=True, rec=True)
        results = parse_res(raw_result)
    except Exception:
        results = []

    # 2. If detection returned no text (common on tight bounding-box crops), run direct recognition (det=False)
    if not results:
        try:
            rec_result = ocr.ocr(image, det=False, rec=True)
            results = parse_res(rec_result)
        except Exception:
            results = []

    return results


class OCRHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass  # Suppress request logging noise

    def do_POST(self):
        if self.path != '/ocr':
            self.send_response(404); self.end_headers(); return

        try:
            length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(length)
            data = json.loads(body)

            # Image is sent as base64-encoded PNG/JPG
            img_bytes = base64.b64decode(data['image'])
            nparr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if img is None:
                raise ValueError("Could not decode image")

            results = run_ocr(img)

            response = json.dumps({"results": results}).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', len(response))
            self.end_headers()
            self.wfile.write(response)

        except Exception as e:
            logging.error(f"OCR error: {e}\n{traceback.format_exc()}")
            err = json.dumps({"results": [], "error": str(e)}).encode()
            self.send_response(500)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', len(err))
            self.end_headers()
            self.wfile.write(err)


if __name__ == '__main__':
    server = HTTPServer(('127.0.0.1', 5001), OCRHandler)
    logging.info("PaddleOCR server listening on http://127.0.0.1:5001")
    server.serve_forever()
