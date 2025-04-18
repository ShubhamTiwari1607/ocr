//required changes -> make it compatible with indian number plate system

from ultralytics import YOLO
import cv2
import os
import pytesseract
import re
import numpy as np
import requests
from datetime import datetime

# Initialize YOLO model
model = YOLO("best.pt")  # Replace with your trained model path

# Tesseract OCR configuration
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
custom_config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'

# API configuration
API_URL = "http://localhost:8080/api/vehicles/entry"
GATE_NUMBER = 3

# Confidence threshold
MIN_CONFIDENCE = 0.6

# Valid Indian state codes
valid_state_codes = {
    "DL", "UP", "HR", "MH", "KA", "TN", "MP", "RJ", "PB", "GJ", "WB", "AP", "TS", 
    "BR", "CG", "GA", "HP", "JK", "KL", "MN", "ML", "MZ", "NL", "OD", "PY", "SK", 
    "TR", "UK", "JH", "AS"
}

def clean_text(text):
    # Step 1: Strip and sanitize text
    text = text.upper().strip()
    text = re.sub(r'[^A-Z0-9]', '', text)

    # Step 2: Common OCR mistake fixes
    if text.startswith("IDL"):
        text = "DL" + text[3:]
    if text.startswith("1DL"):
        text = "DL" + text[3:]
    if text.startswith("0L"):
        text = "DL" + text[2:]
    if text.startswith("L"):
        text = "DL" + text[1:]
    
    # Step 3: Attempt to match valid Indian plate pattern
    # Expected: State(2L) + RTO(1-2D) + Series(1-3L) + Number(3-4D)
    match = re.search(r'([A-Z]{2})(\d{1,2})([A-Z]{1,3})(\d{3,4})', text)
    if match:
        plate = ''.join(match.groups())
        if plate[:2] in valid_state_codes:
            return plate

    # Step 4: Handle jumbled formats like 12KAAB1234
    match = re.search(r'(\d{1,2})([A-Z]{2})([A-Z]{1,3})(\d{3,4})', text)
    if match:
        state, rto, series, number = match.group(2), match.group(1), match.group(3), match.group(4)
        if state in valid_state_codes:
            return f"{state}{rto}{series}{number}"

    return text

def detect_plate_color(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    avg_color_per_row = np.average(hsv, axis=0)
    avg_hsv = np.average(avg_color_per_row, axis=0)
    hue, sat, val = avg_hsv

    if val > 180:
        if hue < 30:
            return "Yellow"
        else:
            return "White"
    elif hue < 15 or hue > 165:
        return "Red"
    elif 35 < hue < 85:
        return "Green"
    elif 85 < hue < 130:
        return "Blue"
    else:
        return "Unknown"

def determine_vehicle_type(plate_color):
    color_mapping = {
        "White": "PRIVATE",
        "Yellow": "COMMERCIAL",
        "Green": "ELECTRIC",
        "Blue": "GOVERNMENT",
        "Red": "TEMPORARY",
        "Unknown": "UNKNOWN"
    }
    return color_mapping.get(plate_color, "UNKNOWN")

def process_license_plate_image(plate_image):
    gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (0, 0), fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    text = pytesseract.image_to_string(morph, config=custom_config)
    
    print("Raw OCR Text:", text)
    return clean_text(text)

def send_vehicle_data(vehicle_number, vehicle_type, image_name, timestamp):
    headers = {"Content-Type": "application/json"}
    payload = {
        "vehicleNumber": vehicle_number,
        "entryGate": GATE_NUMBER,
        "vehicleType": vehicle_type,
        "imageName": image_name,
        "timestamp": timestamp,
    }

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = requests.post(API_URL, json=payload, headers=headers, timeout=5)
            if response.status_code in (200, 201):
                print(f"✅ Successfully sent data for {vehicle_number}")
                return True
            else:
                print(f"❌ Attempt {attempt + 1}: Failed to send data - {response.status_code}")
        except Exception as e:
            print(f"⚠️ Attempt {attempt + 1}: Error sending data - {str(e)}")
    print(f"❌ Failed to send data for {vehicle_number} after {max_retries} attempts")
    return False

def detect_and_process_vehicles(video_source, output_dir="output"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"Error: Unable to open video source at {video_source}")
        return

    frame_skip = 2
    frame_count = 0
    detection_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % frame_skip != 0:
            continue

        enhanced_frame = cv2.detailEnhance(frame, sigma_s=10, sigma_r=0.15)
        results = model(enhanced_frame, conf=MIN_CONFIDENCE, imgsz=640)

        for result in results:
            for box, confidence in zip(result.boxes.xyxy, result.boxes.conf):
                x1, y1, x2, y2 = map(int, box)

                pad = 5
                x1 = max(0, x1 - pad)
                y1 = max(0, y1 - pad)
                x2 = min(enhanced_frame.shape[1], x2 + pad)
                y2 = min(enhanced_frame.shape[0], y2 + pad)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{confidence:.2f}", (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                plate_img = enhanced_frame[y1:y2, x1:x2]
                if plate_img.size < 500:
                    continue

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                output_path = os.path.join(output_dir, f"plate_{timestamp}_{detection_count}.jpg")
                cv2.imwrite(output_path, plate_img)

                plate_text = process_license_plate_image(plate_img)
                plate_color = detect_plate_color(plate_img)
                vehicle_type = determine_vehicle_type(plate_color)

                print(f"📸 Plate: {plate_text} | Confidence: {confidence:.2f} | Color: {plate_color} | Type: {vehicle_type}")

                if len(plate_text) >= 6 and plate_text[:2] in valid_state_codes:
                    send_vehicle_data(plate_text, vehicle_type, os.path.basename(output_path), timestamp)

                detection_count += 1

        cv2.imshow('Multi-Plate Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"✅ Processing completed. Total plates detected: {detection_count}")

if __name__ == "__main__":
    VIDEO_SOURCE = "mycarplate.mp4"  # Or local file like "video.mp4"
    detect_and_process_vehicles(VIDEO_SOURCE)
