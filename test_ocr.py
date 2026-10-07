import sys
import os
import subprocess

# PaddlePaddle requires Python <= 3.12. If running under Python 3.13+, re-exec with Python 3.12.
if sys.version_info >= (3, 13):
    print(f"Detected Python {sys.version_info.major}.{sys.version_info.minor}. PaddleOCR requires Python 3.12.")
    print("Re-launching script under Python 3.12 ('py -3.12')...")
    try:
        py312 = subprocess.check_output(["py", "-3.12", "-c", "import sys; print(sys.executable)"]).decode().strip()
        ret = subprocess.call([py312] + sys.argv)
        sys.exit(ret)
    except Exception as err:
        print(f"Error re-launching under Python 3.12: {err}", file=sys.stderr)
        print("Please run using: py -3.12 test_ocr.py", file=sys.stderr)
        sys.exit(1)

import cv2
from paddleocr import PaddleOCR

def test_ocr():
    print("Loading OCR...")
    ocr = PaddleOCR(use_textline_orientation=False, lang='en', device='cpu', enable_mkldnn=False)
    
    # We will just pass a dummy image
    import numpy as np
    img = np.ones((100, 300, 3), dtype=np.uint8) * 255
    cv2.putText(img, "220926 028A", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.putText(img, "10:35", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    
    print("Running inference...")
    result = ocr.ocr(img)
    print("Type of result:", type(result))
    
    # Convert to list if it's a generator
    if hasattr(result, '__iter__') and not isinstance(result, list):
        result = list(result)
        
    print("Result:")
    for i, r in enumerate(result):
        print(f"[{i}] type={type(r)} -> {r}")
        
    print("Dict properties if applicable:")
    try:
        for r in result:
            if hasattr(r, 'keys'):
                print(r.keys())
            if hasattr(r, '__dict__'):
                print(r.__dict__)
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    test_ocr()
