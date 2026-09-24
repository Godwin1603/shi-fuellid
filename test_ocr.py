import cv2
import sys
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
