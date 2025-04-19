import numpy as np
import cv2
from scipy.stats import skew, kurtosis, entropy
import glob
import os

def features_kappa(img):
    """
    Calculates a 23-dimensional feature set based on Pearson parameter κ.
    """
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Normalize pixel values to range [0, 1]
    img = img.astype(np.float32) / 255.0

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
            sko.append(skew(block))
            kuo.append(kurtosis(block))

    # Filter out NaN values
    sko = [val for val in sko if not np.isnan(val)]
    kuo = [val for val in kuo if not np.isnan(val)]

    sko, kuo = np.array(sko), np.array(kuo)
    kappa = (sko**2) / kuo
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

# ========================== MAIN SCRIPT ===========================

# Path to the folder containing original TIF images
original_images_path = "/home/mallarapuhemavarshini/Desktop/Mystudies/dip/UCID1338"  # <-- Replace if needed

# Find all .tif and .TIF files (case-insensitive)
original_images = glob.glob(os.path.join(original_images_path, '*.tif')) + \
                  glob.glob(os.path.join(original_images_path, '*.TIF'))

# Check how many images were found
print(f"Found {len(original_images)} original TIF images.")
if len(original_images) == 0:
    print("No TIF images found. Please check the path.")

# Process each image
for img_path in sorted(original_images):
    # Load the image
    img = cv2.imread(img_path)

    # Check if image loaded successfully
    if img is None:
        print(f"Failed to load image: {img_path}")
        continue

    try:
        # Extract features
        features = features_kappa(img)
        print(f"Features for {os.path.basename(img_path)}: {features[:10]}...")  # Display first 10 features
    except Exception as e:
        print(f"Error processing {img_path}: {e}")

