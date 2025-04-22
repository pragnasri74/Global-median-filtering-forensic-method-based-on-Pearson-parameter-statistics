import os
import cv2
import numpy as np
from features_kappa import features_kappa
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Paths for original and filtered images
original_images_path = 'UCID1338'
filtered_images_path = 'UCID1338_MedianFiltered'

# Ensure the folders exist
if not os.path.exists(original_images_path):
    raise FileNotFoundError(f"Original images folder not found: {original_images_path}")
if not os.path.exists(filtered_images_path):
    raise FileNotFoundError(f"Filtered images folder not found: {filtered_images_path}")

# Process only the first 20 images (assuming filenames are 1.tif, 2.tif, ..., 20.tif)
image_indices = list(range(1, 21))  # From 1 to 20

features = []
labels = []

for idx in image_indices:
    filename = f"{idx}.tif"
    orig_path = os.path.join(original_images_path, filename)
    filt_path = os.path.join(filtered_images_path, filename)

    if not os.path.exists(orig_path):
        print(f"Original image not found: {orig_path}")
        continue
    if not os.path.exists(filt_path):
        print(f"Filtered image not found: {filt_path}")
        continue

    # Load and extract features from original image
    orig_img = cv2.imread(orig_path, cv2.IMREAD_GRAYSCALE)
    if orig_img is not None:
        features.append(features_kappa(orig_img))
        labels.append(0)
    else:
        print(f"Failed to read original image: {orig_path}")

    # Load and extract features from filtered image
    filt_img = cv2.imread(filt_path, cv2.IMREAD_GRAYSCALE)
    if filt_img is not None:
        features.append(features_kappa(filt_img))
        labels.append(1)
    else:
        print(f"Failed to read filtered image: {filt_path}")

# Convert to NumPy arrays
features = np.array(features)
labels = np.array(labels)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.5, random_state=42)

# Train the SVM classifier
svm = SVC(kernel='rbf', C=1.0, gamma='scale')
svm.fit(X_train, y_train)

# Evaluate the classifier
y_pred = svm.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy using 20 images: {accuracy * 100:.2f}%")

