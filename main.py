import os
from PIL import Image
from vector_emb import get_best_match
from object_crop import crop_and_save_objects
from prompt_chain import multi_level_cot
from draw_box import draw_box_on_image
from blip2.blip2_inference import generate_features
import ast


def main():
    path = input("enter the path of the image: ")
    output_dir = "cropped_objects"
    # os.makedirs(output_dir, exist_ok=True)
    crop_and_save_objects(path, output_dir)
    prompt = input("Prompt: ")
    required_desc = multi_level_cot(prompt)
    folder_path = "./cropped_objects"

    # Iterate through the folder
    captions = {}
    if os.path.exists(folder_path) and os.path.isdir(folder_path):
        for filename in os.listdir(folder_path):
            if filename.lower().endswith(".jpg"):
                relative_path = os.path.join(folder_path, filename)
                text_desc = generate_features(relative_path)
                num = filename[filename.find("_") + 1 : filename.find(".jpg")]
                captions[text_desc] = num

    print(captions)
    req_ind = get_best_match(list(captions.keys()), required_desc)
    data = open("coordinates.txt")
    coord_dict = ast.literal_eval(data.read())
    data.close()
    coordinates = coord_dict[captions[req_ind]]
    draw_box_on_image(path, coordinates, "selected.jpeg")


if __name__ == "__main__":
    main()
