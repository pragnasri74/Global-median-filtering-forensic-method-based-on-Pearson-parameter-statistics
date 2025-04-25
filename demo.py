import os
import cv2
import numpy as np
from features_kappa_f import features_kappa
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score



# Paths for original and filtered images
original_images_path = 'original'  # Folder with original images
filtered_images_path = 'filtered'  # Folder with median-filtered images

# Ensure the folders exist
if not os.path.exists(original_images_path):
    raise FileNotFoundError(f"Original images folder not found: {original_images_path}")
if not os.path.exists(filtered_images_path):
    raise FileNotFoundError(f"Filtered images folder not found: {filtered_images_path}")

# Fetch all image file names
original_images = [f for f in os.listdir(original_images_path) if f.endswith(('.jpg', '.png', '.tif', '.bmp'))]

filtered_images = [f for f in os.listdir(filtered_images_path) if f.endswith(('.jpg', '.png', '.tif', '.bmp'))]

# Ensure there is a 1-to-1 correspondence between original and filtered images
if len(original_images) != len(filtered_images):
    raise ValueError("The number of original images does not match the number of filtered images.")

# Feature extraction
features = []
labels = []  # 0 for original, 1 for filtered
c=0;
# Extract features for original images
for img_name in original_images:
    img_path = os.path.join(original_images_path, img_name)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    print("good night")
    features.append(features_kappa(img))
    print("good morning")
    c=c+1
    print(c)
    labels.append(0)

# Extract features for filtered images
for img_name in filtered_images:
    img_path = os.path.join(filtered_images_path, img_name)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    
    features.append(features_kappa(img))
    
    labels.append(1)

# Convert features and labels to numpy arrays
features = np.array(features)
labels = np.array(labels)
print("Helooo")
# Train-test split
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.5, random_state=42)

# Train the SVM classifier
svm = SVC(kernel='rbf', C=1.0, gamma='scale')
svm.fit(X_train, y_train)

# Evaluate the classifier
y_pred = svm.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

