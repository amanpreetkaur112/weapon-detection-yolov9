import os

model_path = "/Users/apree/PycharmProjects/Aman45/aman-weapon_detection/weights/best.pt"  # Change for different models
size = os.path.getsize(model_path) / (1024 * 1024)  # Convert bytes to MB

print(f"Model Size: {size:.2f} MB")

# python script to get the size of the trained model.. not needed its a "just-in-case" program