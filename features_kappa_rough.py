import numpy as np
import cv2
from scipy.stats import skew, kurtosis
from scipy.stats import entropy
import glob
import os

def features_kappa(img):
    """
    Calculates a 23-dimensional feature set based on Pearson parameter κ.
    """
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Normalize pixel values to range [0, 1]
    img = img.astype(np.float32) / 255.0  # Convert to float32 for compatibility with OpenCV
    
    # Compute Median Filtered Residual (MFR)
    img_median = cv2.medianBlur(img, 3)
    residual = img - img_median
    
    # Sliding window for 3×3 blocks
    win_sz = 3
    stride = 1
    rows, cols = residual.shape
    sko, kuo = [], []
    for i in range(0, rows - win_sz + 1, stride):
      for j in range(0, cols - win_sz + 1, stride):
        block = residual[i:i + win_sz, j:j + win_sz].flatten()
        if np.var(block) > 1e-6:  # Skip nearly constant blocks
            sko.append(skew(block))
            kuo.append(kurtosis(block))
        else:
            sko.append(0.0)
            kuo.append(0.0)

    
    # Filter out NaN values
    sko = [val for val in sko if not np.isnan(val)]
    kuo = [val for val in kuo if not np.isnan(val)]
    
    # Compute Pearson parameter κ
    sko, kuo = np.array(sko), np.array(kuo)
    epsilon = 1e-8
    kappa = (sko**2) / (kuo + epsilon)

    kappa = kappa[~np.isnan(kappa)]  # Remove NaN κ values

    # Generate histogram (16 bins, range [0, 1.6])
    hist, _ = np.histogram(kappa, bins=16, range=(0, 1.6))
    hist = hist / np.sum(hist)  # Normalize histogram

    # Compute statistical features
    mean = np.mean(kappa)
    variance = np.var(kappa)
    skewness = skew(kappa)
    kurt = kurtosis(kappa)
    energy = np.sum(hist**2)
    entropy_val = entropy(hist + 1e-10)  # Avoid log(0)
    max_bin = np.max(hist)

    # Combine features into a 23-dimensional vector
    features = np.concatenate((hist, [mean, variance, skewness, kurt, energy, entropy_val, max_bin]))
    return features

# Path to the dataset directory
dataset_directory = "/home/mallarapuhemavarshini/Desktop/Mystudies/dip" 
 # Replace with your actual directory path

# Find all .tif files in the directory and subdirectories
tif_images = glob.glob(os.path.join(dataset_directory, '**', '*.tif'), recursive=True)

# Check how many .tif images are found
print(f"Found {len(tif_images)} TIF images.")
if len(tif_images) == 0:
    print("No TIF images found. Check the directory path and extension.")

# Iterate through the list of .tif files
for tif_image_path in tif_images:
    # Load the image
    img = cv2.imread(tif_image_path)
    
    # Check if the image was loaded properly
    if img is None:
        print(f"Failed to load image: {tif_image_path}")
    else:
        print(f"Successfully loaded image: {tif_image_path}")
        
        try:
            # Calculate the features
            features = features_kappa(img)
            print(f"Features for {tif_image_path}: {features[:10]}...")  # Print only the first 10 elements
        except Exception as e:
            print(f"Error processing {tif_image_path}: {e}")
if __name__ == "__main__":
    # Path to the dataset directory
    dataset_directory = "/home/mallarapuhemavarshini/Desktop/Mystudies/dip" 

    # Find all .tif files in the directory and subdirectories
    tif_images = glob.glob(os.path.join(dataset_directory, '**', '*.tif'), recursive=True)

    print(f"Found {len(tif_images)} TIF images.")
    if len(tif_images) == 0:
        print("No TIF images found. Check the directory path and extension.")

    for tif_image_path in tif_images:
        img = cv2.imread(tif_image_path)
        
        if img is None:
            print(f"Failed to load image: {tif_image_path}")
        else:
            print(f"Successfully loaded image: {tif_image_path}")
            
            try:
                features = features_kappa(img)
                print(f"Features for {tif_image_path}: {features[:10]}...")
            except Exception as e:
                print(f"Error processing {tif_image_path}: {e}")


