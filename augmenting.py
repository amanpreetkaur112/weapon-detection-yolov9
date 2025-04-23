import os
import cv2
import numpy as np
import albumentations as A
from shutil import copyfile
from collections import defaultdict

# Define paths to your dataset
dataset_path = "/Users/apree/PycharmProjects/Aman45/aman-dataset"
train_images_path = os.path.join("/Users/apree/PycharmProjects/Aman45/aman-dataset", "train", "images")  # Removed redundant "train/images"
train_labels_path = os.path.join("/Users/apree/PycharmProjects/Aman45/aman-dataset", "train", "labels")  # Removed redundant "train/labels"
output_images_path = os.path.join("/Users/apree/PycharmProjects/Aman45/aman-dataset", "train", "augmented_images")  # Adjusted to keep augmented data in the same dataset folder
output_labels_path = os.path.join("/Users/apree/PycharmProjects/Aman45/aman-dataset", "train", "augmented_labels")  # Adjusted to keep augmented data in the same dataset folder

os.makedirs(output_images_path, exist_ok=True)
os.makedirs(output_labels_path, exist_ok=True)

# ========== AUGMENTATION PIPELINE ========== #
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.3),
    A.Rotate(limit=30, p=0.5),
    A.RandomScale(scale_limit=0.2, p=0.5),
    A.GaussNoise(p=0.3),
    A.Affine(translate_percent=0.1, scale=(0.9, 1.1), rotate=15, p=0.5),
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

# ========== HELPER FUNCTIONS ========== #
def read_yolo_labels(label_path):
    bboxes, class_labels = [], []
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue  # Skip malformed lines
                try:
                    class_id = int(parts[0])
                    x_center, y_center, width, height = map(float, parts[1:5])
                    bboxes.append([x_center, y_center, width, height])
                    class_labels.append(class_id)
                except ValueError:
                    continue
    return bboxes, class_labels

def write_yolo_labels(label_path, bboxes, class_labels):
    with open(label_path, 'w') as f:
        for bbox, class_id in zip(bboxes, class_labels):
            f.write(f"{class_id} {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}\n")

def get_image_path(image_name, images_path):
    for ext in ['.jpg', '.jpeg', '.png']:
        image_path = os.path.join(images_path, f"{image_name}{ext}")
        if os.path.exists(image_path):
            return image_path
    return None

# ========== STEP 1: COPY ORIGINAL IMAGES ========== #
print("Copying original files...")
for label_file in os.listdir(train_labels_path):
    if label_file.endswith('.txt'):
        image_name = label_file.rsplit('.', 1)[0]
        image_path = get_image_path(image_name, train_images_path)
        if image_path:
            # Copy image
            output_image_path = os.path.join(output_images_path, os.path.basename(image_path))
            copyfile(image_path, output_image_path)
            # Copy label
            label_path = os.path.join(train_labels_path, label_file)
            output_label_path = os.path.join(output_labels_path, label_file)
            copyfile(label_path, output_label_path)

# ========== STEP 2: COUNT ORIGINAL CLASS DISTRIBUTION ========== #
class_counts = defaultdict(int)
print("\nCounting original class distribution...")
for label_file in os.listdir(train_labels_path):
    bboxes, class_labels = read_yolo_labels(os.path.join(train_labels_path, label_file))
    for class_id in class_labels:
        class_counts[class_id] += 1

print("Original class counts:", dict(class_counts))

# ========== STEP 3: AUGMENT TO REACH 3000 PER CLASS ========== #
TARGET_IMAGES_PER_CLASS = 3000
print("\nStarting augmentation...")

# Track augmented counts separately (to avoid overcounting)
augmented_counts = defaultdict(int)

for class_id in class_counts:
    current_count = class_counts[class_id]
    if current_count >= TARGET_IMAGES_PER_CLASS:
        print(f"Class {class_id} already has enough samples ({current_count})")
        continue

    needed = TARGET_IMAGES_PER_CLASS - current_count
    print(f"Augmenting class {class_id} (needed: {needed})")

    # Find all images containing this class
    class_images = []
    for label_file in os.listdir(train_labels_path):
        bboxes, class_labels = read_yolo_labels(os.path.join(train_labels_path, label_file))
        if class_id in class_labels:
            class_images.append(label_file)

    # Augment until we reach the target
    while augmented_counts[class_id] < needed:
        chosen_label = np.random.choice(class_images)
        image_name = chosen_label.rsplit('.', 1)[0]
        image_path = get_image_path(image_name, train_images_path)
        if not image_path:
            continue

        # Read image and labels
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        bboxes, class_labels = read_yolo_labels(os.path.join(train_labels_path, chosen_label))

        # Apply augmentation
        try:
            augmented = transform(image=image, bboxes=bboxes, class_labels=class_labels)
            aug_image = augmented['image']
            aug_bboxes = augmented['bboxes']
            aug_class_labels = augmented['class_labels']

            # Filter invalid boxes
            valid_bboxes = []
            valid_classes = []
            for bbox, cls in zip(aug_bboxes, aug_class_labels):
                x, y, w, h = bbox
                if (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
                    valid_bboxes.append(bbox)
                    valid_classes.append(cls)

            if not valid_bboxes:
                continue  # Skip if no valid boxes

            # Save augmented data
            ext = os.path.splitext(image_path)[1]
            aug_image_name = f"aug_{class_id}_{augmented_counts[class_id]}{ext}"
            aug_label_name = f"aug_{class_id}_{augmented_counts[class_id]}.txt"

            aug_image_path = os.path.join(output_images_path, aug_image_name)
            aug_label_path = os.path.join(output_labels_path, aug_label_name)

            cv2.imwrite(aug_image_path, cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR))
            write_yolo_labels(aug_label_path, valid_bboxes, valid_classes)

            augmented_counts[class_id] += 1  # Track augmentation progress

        except Exception as e:
            print(f"Error augmenting {image_path}: {e}")
            continue

# ========== STEP 4: VERIFY FINAL COUNTS ========== #
print("\nVerifying final counts...")
final_class_counts = defaultdict(int)
for label_file in os.listdir(output_labels_path):
    bboxes, class_labels = read_yolo_labels(os.path.join(output_labels_path, label_file))
    for class_id in class_labels:
        final_class_counts[class_id] += 1

print("Final class counts after augmentation:", dict(final_class_counts))
print("\nAugmentation complete! ✅")