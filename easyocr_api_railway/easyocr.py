from flask import Flask, request, jsonify
from ultralytics import YOLO
from paddleocr import PaddleOCR
import cv2
import numpy as np
import base64
import os

app = Flask(__name__)

# Load models once at startup
model = YOLO("best.pt")
ocr = PaddleOCR(use_angle_cls=True, lang='ar', show_log=False)

def decode_base64_image(b64: str):
    try:
        data = base64.b64decode(b64)
        np_arr = np.frombuffer(data, np.uint8)
        return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    except Exception as e:
        app.logger.error(f"Image decode error: {str(e)}")
        return None

def predict_license_plates(img):
    results = model(img)[0]
    extracted_texts = []

    for box in results.boxes:
        x_min, y_min, x_max, y_max = map(int, box.xyxy[0].tolist())
        cropped_img = img[y_min:y_max, x_min:x_max]

        # Image processing
        hsv = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2HSV)
        # ... (keep your existing color masking logic)

        # OCR processing
        ocr_results = ocr.ocr(result)
        extracted_text = ""
        
        if ocr_results:
            for line in ocr_results:
                for word_info in line:
                    text = word_info[1][0]
                    extracted_text += text + " "
        
        extracted_texts.append(extracted_text[::-1].strip())

    return extracted_texts

@app.route("/predict", methods=["POST"])
def predict():
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400
        
    data = request.get_json()
    img = decode_base64_image(data.get("image", ""))
    
    if img is None:
        return jsonify({"error": "Invalid image data"}), 400
    
    try:
        plates = predict_license_plates(img)
        return jsonify({"predictions": plates}) if plates else jsonify({"error": "No plates found"}), 200
    except Exception as e:
        app.logger.error(f"Prediction error: {str(e)}")
        return jsonify({"error": "Processing failed"}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
