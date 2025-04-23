import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO

def detect_weapons(image_path, model_path='/Users/apree/PycharmProjects/Aman45/aman-weapon_detection/weights/best.pt'):
    # 1. Load trained YOLOv9m model
    model = YOLO(model_path)

    # 2. Read input image
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Error: Could not read image {image_path}")
        return

    # 3. Run inference
    results = model(frame, stream=True)

    # 4. Process and draw results
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()
        confidences = result.boxes.conf.cpu().numpy()

        for box, cls, conf in zip(boxes, classes, confidences):
            x1, y1, x2, y2 = map(int, box)
            class_name = model.names[int(cls)]

            # Draw rectangle and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{class_name} {conf:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Convert BGR (OpenCV) to RGB (matplotlib)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Use matplotlib to display the image
    plt.imshow(frame_rgb)
    plt.axis('off')
    plt.title('Weapon Detection')
    plt.show()


# Usage
detect_weapons('testingimg/test4 N.jpeg')  # Path to your test image