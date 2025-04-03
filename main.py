import subprocess
import os
import requests
import re

# Get the absolute path of the current directory
script_dir = os.path.dirname(os.path.abspath(__file__))

# Paths to detect.py and extract_number.py
detect_script = os.path.join(script_dir, "detect.py")
extract_script = os.path.join(script_dir, "extract_number.py")

# Run detect.py
print("Running detect.py...")
subprocess.run(["python", detect_script])

# Run extract_number.py
print("Running extract_number.py...")
# Capture the output of extract_number.py
result = subprocess.run(["python", extract_script], capture_output=True, text=True)
extracted_text = result.stdout.strip()  # Assuming the script outputs the vehicle number

# Debugging: Print the raw extracted text
print(f"Raw Extracted Text: {extracted_text}")

# Sanitize the extracted text to extract only the vehicle number
VEHICLE_NUMBER_PATTERN = r"[A-Z]{2}\d{2}[A-Z]{1,2}\d{1,4}"  # Example pattern for vehicle numbers
match = re.search(VEHICLE_NUMBER_PATTERN, extracted_text)

if not match:
    print(f"No valid vehicle number found in: {extracted_text}. Exiting.")
    exit()

# Get the matched vehicle number
extracted_vehicle_number = match.group(0)

# Debugging: Print the sanitized vehicle number
print(f"Sanitized Vehicle Number: {extracted_vehicle_number}")

# Maintain a set of vehicles that have been marked for entry
processed_vehicles = set()

# Check if the vehicle has already been marked for entry
if extracted_vehicle_number in processed_vehicles:
    # Mark exit for the vehicle
    exit_gate = 2  # Example exit gate
    exit_url = f"http://localhost:8080/api/vehicles/exit/{extracted_vehicle_number}/{exit_gate}"
    print(f"Marking exit for vehicle: {extracted_vehicle_number} at gate {exit_gate}")
    try:
        exit_response = requests.put(exit_url)
        print(f"Exit Response Status Code: {exit_response.status_code}")
        print(f"Exit Response Text: {exit_response.text}")
        if exit_response.status_code == 200:
            print("Exit time successfully marked!")
        else:
            print("Failed to mark exit time.")
            print("Status Code:", exit_response.status_code)
            print("Response:", exit_response.text)
    except requests.exceptions.RequestException as e:
        print(f"Error occurred while marking exit: {e}")
else:
    # Mark entry for the vehicle
    entry_url = "http://localhost:8080/api/vehicles/entry"
    entry_data = {
        "vehicleNumber": extracted_vehicle_number,
        "entryGate": 3,
        "vehicleType": "PRIVATE",
        "imageName": "vehicle_1234.jpg"
    }

    # Debugging: Print the payload
    print(f"Payload being sent to the backend for entry: {entry_data}")

    print("Sending data to the backend for entry...")
    try:
        entry_response = requests.post(entry_url, json=entry_data)
        print(f"Entry Response Status Code: {entry_response.status_code}")
        print(f"Entry Response Text: {entry_response.text}")

        if entry_response.status_code == 200 or entry_response.status_code == 201:
            print("Entry successfully marked!")
            # Add the vehicle to the processed set
            processed_vehicles.add(extracted_vehicle_number)
        else:
            print("Failed to mark entry.")
            print("Status Code:", entry_response.status_code)
            print("Response:", entry_response.text)
    except requests.exceptions.RequestException as e:
        print(f"Error occurred while marking entry: {e}")

print("Processing completed!")
