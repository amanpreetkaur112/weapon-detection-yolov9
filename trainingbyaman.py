import torch
import cv2
import ssl
from ultralytics import YOLO
import matplotlib.pyplot as plt

# Disable SSL certificate verification (to handle SSL errors)
ssl._create_default_https_context = ssl._create_unverified_context

# Load the model for weapon detection
model_path= '/Users/apree/PycharmProjects/Aman45/aman-weapon_detection/weights/best.pt';
model = YOLO(model_path)

# Print class names for debugging
print("Model class names:", model.names)  # Debug model classes

# Define weapon classes (adjust based on actual model training)
weapon_classes = ["knife", "pistol", "rifle"]

# Load the image
img_path = 'testingimg/img1.jpeg'  # Change this to the path of your image
img = cv2.imread(img_path)

# Perform weapon detection using YOLOv8n
results = model(img)

# Extract and print the weapon names detected
for result in results:
    for box in result.boxes:  # Access detection results
        conf = box.conf.item()  # Confidence score
        cls = int(box.cls.item())  # Class ID
        weapon_name = model.names[cls]

        # Only print detected weapons
        if weapon_name in weapon_classes and conf > 0.3:
            print(f"Detected weapon: {weapon_name} with confidence: {conf:.2f}")

# Overlay detections on the image
annotated_image = results[0].plot()

# Save the output image
cv2.imwrite("output.jpg", annotated_image)
print("Output image saved as output.jpg")

# Display image using matplotlib
plt.imshow(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.show()

# this python script creates a output.jpg which stores the input image along with detected objects