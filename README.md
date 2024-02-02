from google.colab import drive
drive.mount('/content/drive')
from IPython.display import display, Image
import cv2
import numpy as np
import os

# Replace 'your_image_directory' with the path to your image directory.
image_directory = '/content/drive/MyDrive/Project_Test/'
output_directory = 'output_directory'  # Replace with the directory where you want to save processed images.

# Ensure the output directory exists, create it if necessary.
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

# List all image files in the directory.
image_files = [os.path.join(image_directory, file) for file in os.listdir(image_directory) if file.endswith(('.jpg', '.jpeg', '.png'))]

# Process each image.
for image_file in image_files:
    # Load the image.
    img = cv2.imread(image_file)

    if img is not None:
        # Resize the image to a standard size (e.g., 224x224 pixels).
        img = cv2.resize(img, (224, 224))

        # Normalize pixel values to the range [0, 255].
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)

        # Apply Gaussian blur for noise reduction.
        img = cv2.GaussianBlur(img, (5, 5), 0)

        # Convert the OpenCV image to a format that can be displayed in Colab.
        _, img_encoded = cv2.imencode('.png', img)
        img_display = Image(data=img_encoded)

        # Show the processed image.
        display(img_display)

        # Save the processed image to the output directory.
        output_file = os.path.join(output_directory, os.path.basename(image_file))
        cv2.imwrite(output_file, img)

print(f"Processed {len(image_files)} images and saved them in {output_directory}.")
