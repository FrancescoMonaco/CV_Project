Folders and files descriptions:
- 1 Train-Detect, folder with the python code for the SVM to detect the players
- 2 Classify-Segment-Eval, folder with the C++ code for all the other tasks
- Images, folder with the original images
- Masks, folder with the original results 
- Predictions, folder with the .txt bounding boxes, and .png segmentations
- ProcessedBoxes, intermediate folder with partial results from python
- Train, folder with the dataset to train the SVM
	-subfolder Positive, contains the positive examples; similarly Negative
- Extra, a folder with a new test dataset, results provided in the relative subfolder
	show the robustness of the methods developed
- CV_Project-Monaco_Russo.pdf, report with all results and explanations

Usage:
Execute 1 Train-Detect, it takes about 17 minutes, a ReadMe in the folder will explain how to run it
Execute 2 Classify-Segment-Eval, it takes about 5 minutes, a ReadMe in the folder will explain how to run it

IMPORTANT NOTICE:
The path to the folder "Project_Monaco_Russo" MUSTN'T contain numbers, this because some methods parse the string 
to find the number of the image in order to locate the same file where 
the boxes and the segmentations should be stored
 e.g. "Path/to/folder/Project_Monaco_Russo"