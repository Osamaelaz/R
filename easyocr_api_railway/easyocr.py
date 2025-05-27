from flask import Flask, request, jsonify
from ultralytics import YOLO
import easyocr
import base64
import cv2
import numpy as np

model  = YOLO("best.pt")
reader = easyocr.Reader(['ar'])

def decode_base64_image(b64: str):
    try:
        data   = base64.b64decode(b64)
        np_arr = np.frombuffer(data, np.uint8)
        return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    except Exception:
        return None

def predict_license_plates(img):
    results = model(img)[0]
    plates  = []
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        y1 += 70
        cropped = img[y1:y2, x1:x2]
        plates += reader.readtext(cropped, detail=0)
    return plates

app = Flask(__name__)
@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json(force=True)
    img     = decode_base64_image(payload.get("image", ""))
    print(img)
    if img is None:
        return jsonify({"error": "Invalid image"}), 400

    plates = predict_license_plates(img)
    if not plates:
        return jsonify({"error": "No text found"}), 200

    return jsonify({"prediction": plates[0]})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
