import cv2
import pytesseract
import os
import re
import numpy as np

# Set Tesseract path
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

current_directory = os.path.dirname(os.path.abspath(__file__))  # Get script's directory

# Folder containing detected number plate images
image_folder = os.path.join(current_directory, "output")
if not os.path.exists(image_folder):
    print(f"Error: The folder '{image_folder}' does not exist!")
    exit()

# OCR Config (Uses PSM 6 for better block text detection)
custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'

# ✅ List of valid Indian state codes
valid_state_codes = {
    "DL", "UP", "HR", "MH", "KA", "TN", "MP", "RJ", "PB", "GJ", "WB", "AP", "TS", "BR", "CG",
    "GA", "HP", "JK", "KL", "MN", "ML", "MZ", "NL", "OD", "PY", "SK", "TR", "UK", "JH", "AS"
}

# Function to clean and correct OCR mistakes
def clean_text(text):
    text = text.strip()
    text = re.sub(r'[^A-Z0-9]', '', text)  # Remove unwanted symbols
    return text  # Keep this simple for now

# Function to detect number plate color
def detect_plate_color(image):
    avg_color_per_row = np.average(image, axis=0)
    avg_color = np.average(avg_color_per_row, axis=0)

    # Convert BGR to RGB
    r, g, b = avg_color[2], avg_color[1], avg_color[0]

    # Adjust thresholds for more accurate color detection
    if r > 180 and g > 180 and b > 180:
        return "White"
    elif r > 180 and g < 120 and b < 120:
        return "Red"
    elif r < 120 and g > 180 and b < 120:
        return "Green"
    elif r < 120 and g < 120 and b > 180:
        return "Blue"
    elif r > 180 and g > 180 and b < 120:
        return "Yellow"
    else:
        return "Unknown"


# Process images in the folder
for image_name in os.listdir(image_folder):
    image_path = os.path.join(image_folder, image_name)

    # Read image
    img = cv2.imread(image_path)

    # Detect plate color
    plate_color = detect_plate_color(img)

    # Resize for better OCR (scaling up)
    img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    # Convert to Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Unsharp Mask (Sharpening)
    sharp = cv2.GaussianBlur(gray, (0, 0), 3)
    sharpened = cv2.addWeighted(gray, 1.5, sharp, -0.5, 0)

    # Apply Bilateral Filtering (Removes noise but keeps edges)
    filtered = cv2.bilateralFilter(sharpened, 9, 75, 75)

    # Apply Otsu's Thresholding (Better for printed text)
    _, thresh = cv2.threshold(filtered, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Debug: Save the preprocessed image to check OCR input
    debug_path = os.path.join(current_directory, f"debug_{image_name}")
    cv2.imwrite(debug_path, thresh)

    # Use Tesseract OCR
    extracted_text = pytesseract.image_to_string(thresh, config=custom_config)

    # Clean and correct extracted text
    cleaned_text = clean_text(extracted_text)

    # ✅ FINAL BRUTE-FORCE FIX: REPLACE "IDL" WITH "DL"
    if cleaned_text.startswith("IDL"):
        cleaned_text = "DL" + cleaned_text[3:]  # Replace "IDL" with "DL"

    print(f"Extracted Text from {image_name}: {cleaned_text}")
    print(f"Detected Plate Color: {plate_color}")
