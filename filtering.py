import os
from PIL import Image
import numpy as np
from scipy.ndimage import median_filter

# Directory containing your UCID dataset
input_dir = 'UCID1338'  # Replace with your path
output_dir = 'UCID1338_MedianFiltered'  # Replace with your path

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Iterate through the images in the directory
for filename in os.listdir(input_dir):
    if filename.endswith('.tif'):
        img_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, f"median_{filename}")

        # Open image
        image = Image.open(img_path)
        
        # Convert image to numpy array
        img_array = np.array(image)
        
        # Apply median filter (kernel size can be adjusted as needed)
        filtered_img = median_filter(img_array, size=3)  # size is the size of the filter window
        
        # Convert filtered image back to PIL format
        filtered_image = Image.fromarray(filtered_img)
        
        # Save the filtered image
        filtered_image.save(output_path)

        print(f"Processed {filename} and saved to {output_path}")

