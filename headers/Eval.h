#ifndef EVAL_H
#define EVAL_H
// Eval.h : Francesco Pio Monaco
#include <string>
#include <vector>
#include <opencv2/core.hpp>

struct BoundingBox {
    int fileNum;
    int x1;
    int y1;
    int width;
    int height;
    int id;
};

/// @brief Returns a vector of bounding boxes from the given path
/// @param filePath , path to the file containing the bounding box data
/// @param hasID , true if the file contains the id of the bounding box, false otherwise
/// @param reverse , true if the id needs to be reversed, false otherwise
/// @return vector of bounding boxes
std::vector<BoundingBox> loadBoundingBoxData(const std::string& filePath, bool hasID = true, bool reverse = false);

/// @brief Returns a vector with the semantic segmentation images
/// @param filePath path to the file containing the data
/// @param reverse , true if the id needs to be reversed, false otherwise
/// @return a vector of Mat
std::vector<cv::Mat> loadSemanticSegmentationData(const std::string& filePath, bool reverse = false);

/// @brief Computes the mAP over the predictions and the golden data
/// @param resultData , vector of bounding boxes from the result file
/// @param predData , vector of bounding boxes from the prediction file
/// @return mAP
float processBoxPreds(const std::vector<BoundingBox>& resultData, const std::vector<BoundingBox>& predData);

/// @brief computes the AP
/// @param PredData , vector of bounding boxes from the prediction file
/// @param ResultData , vector of bounding boxes from the result file
/// @param label , class label
/// @return AP for given class
float computeAP(const std::vector<BoundingBox> PredData, const std::vector<BoundingBox> ResultData, int label);

/// @brief computes the AP for a single image and prints it
/// @param resultData , vector of bounding boxes from the result file
/// @param resultData_rev , vector of bounding boxes from the result file reversed
/// @param predData , vector of bounding boxes from the prediction file
/// @param size , size of the vector
void singleImageAP(const std::vector<BoundingBox> resultData, const std::vector<BoundingBox> resultData_rev, const std::vector<BoundingBox> predData, int size);

/// @brief Computes IoU for the segmentation task over the datasets provided
/// @param resultData , vector of gold segmentation images
/// @param predData , vector of predicted segmentation images 
/// @return IoU value
float processSemanticSegmentation(const std::vector<cv::Mat>& resultData, const std::vector<cv::Mat>& predData);

/// @brief Computes IoU for the detection task over the bounding boxes
/// @param box1 bounding box
/// @param box2 bounding box
/// @return iou
double computeIoUBB(const BoundingBox& box1, const BoundingBox& box2);

/// @brief computes IoU for the couple of Mat
/// @param result , gold Mat
/// @param pred , prediction Mat
/// @return IoU for the couple
float computeIoUSEG(const cv::Mat& result, const cv::Mat& pred);

/// @brief show on screen the results
/// @param source , path for the original images
/// @param path , path for the images
void showResults(const std::string& source, const std::string& path);

/// @brief Extracts the number from the string
/// @param s  
/// @return the number or -1 if no number is found
int extractNumber(const cv::String& s);


#endif // !EVAL_H
