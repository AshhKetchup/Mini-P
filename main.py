import os
from PIL import Image
from vector_emb import get_best_match
from object_crop import crop_and_save_objects
from prompt_chain import multi_level_cot
from blip2.blip2_inference import generate_features

def main():
    path = input("enter the path of the image: ")
    output_dir = "cropped_objects"
    # os.makedirs(output_dir, exist_ok=True)
    crop_and_save_objects(path, output_dir)
    

    required_desc=multi_level_cot("Open A Parcel")

    folder_path = './cropped_objects'

    # Iterate through the folder
    captions = []
    if os.path.exists(folder_path) and os.path.isdir(folder_path):
        for filename in os.listdir(folder_path):
            if filename.lower().endswith('.jpg'):
                relative_path = os.path.join(folder_path, filename)
                text_desc = generate_features(relative_path)
                captions.append(text_desc)

    # text_desc = ["dull","pointy","elongated"]

    print(captions)

    req_ind = get_best_match(captions, required_desc)
    print(req_ind)


if __name__ == "__main__":
    main()
