import os
import cv2
from ultralytics import YOLO

# Initialize the YOLOv8 model
model = YOLO("yolov8m.pt")  # Replace with the path to your YOLOv8 model if custom-trained

# Directory to save cropped objects
output_dir = "cropped_objects"
os.makedirs(output_dir, exist_ok=True)


# Function to crop and save detected objects
def crop_and_save_objects(image_path, output_dir):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error loading image: {image_path}")
        return

    # Run YOLOv8 inference
    results = model(image)

    # Iterate over detected objects
    for idx, box in enumerate(results[0].boxes.xyxy):
        # Extract coordinates
        x1, y1, x2, y2 = map(int, box)  # Convert to integers

        # Crop the object
        cropped_object = image[y1:y2, x1:x2]

        # Save the cropped object
        object_path = os.path.join(output_dir, f"object_{idx + 1}.jpg")
        cv2.imwrite(object_path, cropped_object)
        print(f"Saved: {object_path}")


# Path to the input image
input_image = "test.webp"  # Replace with the path to your image

# Crop and save all objects
crop_and_save_objects(input_image, output_dir)