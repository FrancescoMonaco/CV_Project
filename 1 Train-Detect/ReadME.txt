Libraries needed:
-sklearn, for the SVM, Scaler
-opencv, to process the image, compute HOG
-joblib, to save the model
-imutils, to do non maxima suppression
-imgaug, to do data augmentation
-tqdm, to show a progress bar

Usage:
python main.py <Path/to/the/Project_Monaco_Russo>

Computing time:
about 4 minutes for training, 10 to process all the images

Model saved:
model_SVM.npy contains the model of the SVM