import torch
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO

# Load trained model
model_path= '/Users/apree/PycharmProjects/Aman45/aman-weapon_detection/weights/best.pt';
model = YOLO(model_path)

# Open webcam (0 = default webcam)
cap = cv2.VideoCapture(0)

# Check if webcam is opened
if not cap.isOpened():
    print("Error: Cannot open webcam.")
    exit()

# Loop through webcam frames
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run model on the frame
    results = model(frame)

    # Draw bounding boxes
    for result in results:
        boxes = result.boxes  # Bounding boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0].item()
            cls = int(box.cls[0])
            label = f"{model.names[cls]}: {conf:.2f}"

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Show frame using Matplotlib
    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.pause(0.001)

cap.release()
plt.close()