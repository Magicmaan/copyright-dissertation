import os
import shutil
import random
from typing import List
import zipfile
from PIL import Image


CONTENT_PATH = os.path.join(os.path.dirname(__file__), "content")
STYLE_PATH = os.path.join(os.path.dirname(__file__), "style")


def resize_images_to_square(image_path: str, size: int = 256) -> None:
    assert os.path.exists(image_path), "The provided image path does not exist."
    processed_count = 0
    for root, _, files in os.walk(image_path):
        for file in files:
            if file.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif")):
                file_path = os.path.join(root, file)
                with Image.open(file_path) as img:
                    # Calculate cropping dimensions
                    width, height = img.size
                    min_dim = min(width, height)
                    left = (width - min_dim) // 2
                    top = (height - min_dim) // 2
                    right = left + min_dim
                    bottom = top + min_dim
                    img = img.crop((left, top, right, bottom))
                    img = img.resize((size, size))
                    img.save(file_path)
                    print("Resized and cropped:", file_path)
                processed_count += 1


resize_images_to_square(CONTENT_PATH)
resize_images_to_square(STYLE_PATH)


def limit_files_per_prefix(directory_path: str, max_files_per_prefix: int = 2) -> None:
    assert os.path.exists(directory_path), "The provided directory path does not exist."
    file_groups = {}

    for root, _, files in os.walk(directory_path):
        for file in files:
            prefix = file.split("_")[0]
            if prefix not in file_groups:
                file_groups[prefix] = []
            file_groups[prefix].append(os.path.join(root, file))

    for prefix, file_list in file_groups.items():
        if len(file_list) > max_files_per_prefix:
            file_list.sort()  # Sort files to keep the first ones
            for file_to_delete in file_list[max_files_per_prefix:]:
                os.remove(file_to_delete)
                print("Deleted:", file_to_delete)


limit_files_per_prefix(STYLE_PATH)
