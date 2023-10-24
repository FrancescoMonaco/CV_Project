# -*- coding: utf-8 -*-
"""
Original file is located at
    https://colab.research.google.com/drive/1c6O9ql8V9U63T6KekPDfXmzo0-evDwnI

#Main.py : Monaco Francesco Pio
"""

# Base imports
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from imutils.object_detection import non_max_suppression
import numpy as np
import os, sys, imutils, cv2, joblib
from imgaug import augmenters as iaa
from tqdm import tqdm
from functions import *


# CHECKING NUMBER OF CMD LINE PARAMTERS
assert len(sys.argv) == 2, "Usage: python main.py <relative_path>"
data_path = sys.argv[1]
assert os.path.isdir(data_path)

# Create an HOG descriptor object
winSize = (64, 128)   # window size
blockSize = (16, 16)  # block size
blockStride = (4, 4)  # block stride
cellSize = (8, 8)   # cell size
nbins = 12            # num of bins
hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins)

# Path to images
pos_im_path = data_path + r"/Train/Positive"
neg_im_path= data_path + r"/Train/Negative"

# Read the image files
pos_im_list = os.listdir(pos_im_path)
neg_im_list = os.listdir(neg_im_path)

# Data for the ML system
seq = iaa.Sequential([
    iaa.Fliplr(0.5),                   # Horizontal flipping
    iaa.Affine(rotate=(-10, 10)),      # Rotation
    iaa.GaussianBlur(sigma=(0, 0.5)),  # Gaussian blur
])

data = []
labels = []

for file in pos_im_list:
    img = cv2.imread(pos_im_path + '/' + file, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (64, 128), interpolation=cv2.INTER_AREA)

    # Append the original image's HOG features
    fd_original = hog.compute(img)
    data.append(fd_original.flatten())
    labels.append(1)

    # Augment the image
    augmented_img = seq.augment_image(img)

    fd_augmented = hog.compute(augmented_img)
    data.append(fd_augmented.flatten())
    labels.append(1)

for file in neg_im_list:
    img = cv2.imread(neg_im_path + '/' + file, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (64, 128), interpolation=cv2.INTER_AREA)

    # Append the original image's HOG features
    fd_original = hog.compute(img)
    data.append(fd_original.flatten())
    labels.append(0)

    # Augment the image
    augmented_img = seq.augment_image(img)

    fd_augmented = hog.compute(augmented_img)
    data.append(fd_augmented.flatten())
    labels.append(0)

# Put the strings as labels
le = LabelEncoder()
labels = le.fit_transform(labels)

max_length = max(len(feature) for feature in data)

# Pad HOG feature vectors with zeros to make them consistent (in case of fixed window the pad is 0)
padded_hog_features = []
for feature in data:
    padded_feature = np.pad(feature, (0, max_length - len(feature)), mode='constant')
    padded_hog_features.append(padded_feature)

data = np.array(padded_hog_features, dtype=np.float32)

# Normalize the HOG feature vectors using Min-Max scaling
scaler = MinMaxScaler()

# Construct training/testing split
(trainData, testData, trainLabels, testLabels) = train_test_split(
    data, labels, test_size=0.10, random_state=42)

print("Train data size:", trainData.size)
print("Test data size:", testData.size)

# Fit the scaler on training
trainData = scaler.fit_transform(trainData)

# Define the parameter grid for grid search
param_grid = {
    'C': [0.01, 0.1, 0.5, 1, 10],
    'kernel': ['linear'],
    'gamma': ['scale', 'auto', 0.1, 1, 10]
}

# Classifier
svm = SVC(probability = True, class_weight='balanced')

# Set GridSearch
grid_search = GridSearchCV(svm, param_grid, cv=5)

# Train
print("SVM training started")
grid_search.fit(trainData, trainLabels)

# Print the best hyperparameters found during grid search
print("Best hyperparameters:", grid_search.best_params_)

# Get the best model
best_model = grid_search.best_estimator_

# Evaluate
testData = scaler.transform(testData)
predictions = best_model.predict(testData)
print(classification_report(testLabels, predictions))


# Save the model:
joblib.dump(best_model, 'model_SVM.npy')
model = best_model

# Paths for the work
images_paths = os.listdir(data_path + r'/Images')
images_paths = [file for file in images_paths if file.lower().endswith(".jpg")]
output_dir = data_path + r'/ProcessedBoxes'

for file in tqdm(images_paths, desc="Processing Images"):
  image = cv2.imread(data_path + r'/Images' + '/' + file)

  # Convert the image to grayscale
  gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

  # Initialize a list to store detected bounding boxes
  detected_boxes = []
  boxes_prob = []

  # Get the input dimension of the trained SVM model
  input_dim = model.n_features_in_

  W = gray_image.shape[1]
  H = gray_image.shape[0]
  rects = selective_search(image, method="quality")
  canny_h, canny_w = canny_len(image)

  # Fit the canny length into boxes that of similar size
  if canny_w > 3*canny_h: canny_w=canny_w/3
  if canny_h > 3*canny_w: canny_h= canny_h/3

  # If the canny length is in normal bounds keep the boxes tight
  if canny_h > 350 and canny_w > 500 and canny_w < 600 and canny_h < 400 :
    canny_w = canny_w-100
  else:
  # If too big make it smaller
    if canny_h > 500 : canny_h = canny_h/3
    if canny_w > 500 : canny_w = canny_w/3

  # If too small make it bigger
  if canny_h < 100 : canny_h = canny_h * 2
  if canny_w < 100 : canny_w = canny_w * 2

 # Move through the rectangles
  for (x,y,w,h) in rects:
          window = gray_image[y:y+h,x:x+w]
          # Skip the ones with invalid dimensions

            # If the canny length is in normal bounds keep the boxes very tight
          if canny_h > 230 or (canny_w > 150 and canny_w < 190):
            if w < 0.8 * canny_w or w > canny_w * 2:
              continue
            if h < canny_w * 0.8 or h > canny_w * 2:
              continue
            # Else let them be looser
          else:
            if w < 0.8 * canny_w or w > canny_w * 2.3:
              continue
            if h < canny_w * 0.8 or h > canny_w * 2.3:
              continue
            # Skip the non proportional ones
          if h > 2.3*w or w>2.3*h:
            continue

          scaled = cv2.resize(window, (64,128),  interpolation = cv2.INTER_AREA)
          hog_features = hog.compute(scaled)
          hog_features = scaler.transform(np.array([hog_features]))
                # Make a prediction using the trained SVM
          prediction = model.predict(hog_features)

                # If the prediction is positive (contains a human), store the bounding box
          if prediction == 1:
                    # Convert bounding box coordinates to the original image scale
                    detected_boxes.append((x, y, x+w, y+h))
                    boxes_prob.append(model.predict_proba(hog_features))
  probabilities = [prob[0][1] for prob in boxes_prob]

  # Apply non-maxima suppression to remove overlapping bounding boxes, small images get a lower threshold
  if canny_h < 150 or canny_w < 150:
    detected_boxes = non_max_suppression(np.array(detected_boxes), probs=probabilities, overlapThresh=0.02)
  else:
    detected_boxes = non_max_suppression(np.array(detected_boxes), probs=probabilities, overlapThresh=0.055)



  txt_filename = os.path.splitext(file)[0] + '.txt'
  txt_path = os.path.join(output_dir, txt_filename)

  with open(txt_path, 'w') as f:
    for box in detected_boxes:
        x, y, w, h = box
        f.write(f"{x} {y} {w-x} {h-y}\n")