import os
import cv2
import numpy as np
from multiprocessing import Pool, cpu_count
from features_kappa import features_kappa
from tqdm import tqdm

# --- JPEG compression quality factor for simulation ---
JPEG_QUALITY = 30

def normalize_features(features):
    """
    Linearly scale each feature dimension into the range [-1, +1].
    """
    min_vals = np.min(features, axis=0)
    max_vals = np.max(features, axis=0)
    denom = np.where(max_vals - min_vals == 0, 1, max_vals - min_vals)
    return 2 * (features - min_vals) / denom - 1

def process_image(args):
    """
    Process a single image: read, simulate JPEG compression at Q=30,
    extract features, and return (feature, label).
    """
    file_path, label = args
    try:
        # 1) Read grayscale image
        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Warning: Could not read image {file_path}")
            return None

        # 2) Simulate JPEG compression in-memory
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
        success, buf = cv2.imencode('.jpg', img, encode_param)
        if not success:
            print(f"Warning: JPEG encoding failed for {file_path}")
            return None
        img = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)

        # 3) Feature extraction
        feat = features_kappa(img)
        return (feat, label)

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def main():
    # --- UPDATE THESE PATHS to your image folders ---
    org_path = 'UCID1338'
    mf_path  = 'UCID1338_MedianFiltered'

    # Get sorted file lists
    org_files = sorted(f for f in os.listdir(org_path) if f.endswith('.tif'))
    mf_files  = sorted(f for f in os.listdir(mf_path) if f.endswith('.tif'))

    # Create list of (full_path, label) tuples
    files_labels = []
    for fname in org_files:
        files_labels.append((os.path.join(org_path, fname), 1))
    for fname in mf_files:
        files_labels.append((os.path.join(mf_path, fname), 2))

    # Process images in parallel
    print(f"Processing {len(files_labels)} images using {cpu_count()} workers...")
    with Pool() as pool:
        results = list(tqdm(pool.imap_unordered(process_image, files_labels), total=len(files_labels)))

    # Filter out failed results
    valid_results = [res for res in results if res is not None]
    if not valid_results:
        raise ValueError("No valid images processed!")

    # Separate features and labels
    features, labels = zip(*valid_results)
    features = np.vstack(features)
    labels = np.array(labels).reshape(-1, 1)

    # Normalize and save
    features_norm = normalize_features(features)
    dataset = np.hstack([features_norm, labels])
    np.save('P2_JPEG30_DN128mf3nj30vsmf5nj30U.npy', dataset)
    print(f"Saved dataset with shape {dataset.shape}")

if __name__ == '__main__':
    main()
