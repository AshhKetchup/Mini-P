import os
import cv2
from ultralytics import YOLO

model = YOLO("yolov8m.pt")


output_dir = "cropped_objects"
os.makedirs(output_dir, exist_ok=True)


def crop_and_save_objects(image_path, output_dir):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error loading image: {image_path}")
        return

    results = model(image)
    dict = {}  # store object_path:coordinates
    for idx, box in enumerate(results[0].boxes.xyxy):
        x1, y1, x2, y2 = map(int, box)
        dict[str(idx + 1)] = [x1, y1, x2, y2]
        cropped_object = image[y1:y2, x1:x2]
        object_path = os.path.join(output_dir, f"object_{idx + 1}.jpg")
        cv2.imwrite(object_path, cropped_object)
        print(f"Saved: {object_path}")
    with open("coordinates.txt", "w") as f:
        f.write(str(dict))


# input_image = "bottle.jpeg"
# crop_and_save_objects(input_image, output_dir)
