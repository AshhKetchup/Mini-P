import os
from PIL import Image
from vector_emb import get_best_match
from object_crop import crop_and_save_objects
from prompt_chain import multi_level_cot

def main():
    path = input("enter the path of the image")
    output_dir = "cropped_objects"
    os.makedirs(output_dir, exist_ok=True)
    crop_and_save_objects(path, output_dir)
    
    text_desc = []
    required_desc = ""
    # text_desc = rakim_function()

    # required_desc = om_function()

    required_desc = multi_level_cot("open a parcel")

    req_ind = get_best_match(text_desc, required_desc)
    print(req_ind)


if __name__ == "__main__":
    main()
