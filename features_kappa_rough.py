import numpy as np
import cv2
from scipy.stats import skew, kurtosis
from scipy.stats import entropy

def features_kappa(img):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    img = img.astype(np.float32) / 255.0
    img_median = cv2.medianBlur(img, 3)
    residual = img - img_median

    win_sz = 3
    stride = 1
    rows, cols = residual.shape
    sko, kuo = [], []
    for i in range(0, rows - win_sz + 1, stride):
        for j in range(0, cols - win_sz + 1, stride):
            block = residual[i:i + win_sz, j:j + win_sz].flatten()
            if np.var(block) > 1e-6:
                sko.append(skew(block))
                kuo.append(kurtosis(block))
            else:
                sko.append(0.0)
                kuo.append(0.0)

    sko = [val for val in sko if not np.isnan(val)]
    kuo = [val for val in kuo if not np.isnan(val)]

    sko, kuo = np.array(sko), np.array(kuo)
    epsilon = 1e-8
    kappa = (sko**2) / (kuo + epsilon)
    kappa = kappa[~np.isnan(kappa)]

    hist, _ = np.histogram(kappa, bins=16, range=(0, 1.6))
    hist = hist / np.sum(hist)

    mean = np.mean(kappa)
    variance = np.var(kappa)
    skewness = skew(kappa)
    kurt = kurtosis(kappa)
    energy = np.sum(hist**2)
    entropy_val = entropy(hist + 1e-10)
    max_bin = np.max(hist)

    features = np.concatenate((hist, [mean, variance, skewness, kurt, energy, entropy_val, max_bin]))
    return features

# ONLY RUNS IF CALLED DIRECTLY
if __name__ == "__main__":
    import glob
    import os

    dataset_directory = "/home/mallarapuhemavarshini/Desktop/Mystudies/dip"
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

