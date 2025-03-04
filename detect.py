from ultralytics import YOLO
import cv2
import os

model = YOLO("best.pt")

def detect_number_plate_in_video(video_path, output_dir="output", snapshot_interval=30):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Unable to open video at {video_path}")
        return

    frame_count = 0
    snapshot_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        if frame_count % snapshot_interval == 0:
            results = model(frame)

            for result in results:
                boxes = result.boxes.xyxy
                confidences = result.boxes.conf
                class_ids = result.boxes.cls

                for box, confidence, class_id in zip(boxes, confidences, class_ids):
                    if confidence > 0.5:
                        x1, y1, x2, y2 = map(int, box)

                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                        number_plate = frame[y1:y2, x1:x2]

                        output_path = os.path.join(output_dir, f"number_plate_{snapshot_count}.jpg")
                        cv2.imwrite(output_path, number_plate)
                        print(f"Number plate saved to {output_path}")
                        snapshot_count += 1

            annotated_frame_path = os.path.join(output_dir, f"annotated_frame_{frame_count}.jpg")
            # cv2.imwrite(annotated_frame_path, frame)
            print(f"Annotated frame saved to {annotated_frame_path}")

    cap.release()
    print("Video processing completed.")

if __name__ == "__main__":
    video_path = "mycarplate.mp4"
    detect_number_plate_in_video(video_path)