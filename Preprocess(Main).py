import os
import cv2
import numpy as np

output_dir = "preprocessed_images"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def preprocess_image(image_path, output_path):
    img = cv2.imread(image_path, 0)  # Load the image in grayscale
    height, width = img.shape
    new_width = 1024
    new_height = int((new_width / width) * height)
    img_resized = cv2.resize(img, (new_width, new_height))

    img_normalized = img_resized / 255.0

    img_denoised = cv2.bilateralFilter((img_normalized * 255).astype(np.uint8), d=9, sigmaColor=75, sigmaSpace=75)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_equalized = clahe.apply(img_denoised)

    output_file = os.path.join(output_path, os.path.basename(image_path))
    cv2.imwrite(output_file, img_equalized)
    print(f"Processed and saved: {output_file}")

def preprocess_images_in_folder(input_folder, output_folder):
    for filename in os.listdir(input_folder):
        if filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith(".jpeg"):
            image_path = os.path.join(input_folder, filename)
            preprocess_image(image_path, output_folder)

input_folder = "mammogram_images"
preprocess_images_in_folder(input_folder, output_dir)