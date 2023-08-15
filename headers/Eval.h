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
/// @return vector of bounding boxes
std::vector<BoundingBox> loadBoundingBoxData(const std::string& filePath);

/// @brief Returns a vector with the semantic segmentation images
/// @param filePath path to the file containing the data
/// @return a vector of Mat
std::vector<cv::Mat> loadSemanticSegmentationData(const std::string& filePath);

/// @brief Computes the mAP over the predictions and the golden data
/// @param resultData , vector of bounding boxes from the result file
/// @param predData , vector of bounding boxes from the prediction file
/// @return mAP
float processBoxPreds(const std::vector<BoundingBox>& resultData, const std::vector<BoundingBox>& predData);

/// @brief computes the AP
/// @param precision , vector of precision values for each class at each time point
/// @param recall , vector of recall values for each class at each time point
/// @return AP for each class
std::vector<float> computeAP(const std::vector<std::vector<float>>& precision, const std::vector<std::vector<float>>& recall);

/// @brief Computes IoU for the segmentation task over the datasets provided
/// @param resultData , vector of gold segmentation images
/// @param predData , vector of predicted segmentation images 
/// @return IoU value
float processSemanticSegmentation(const std::vector<cv::Mat>& resultData, const std::vector<cv::Mat>& predData);

/// @brief computes IoU for the couple of Mat
/// @param result , gold Mat
/// @param pred , prediction Mat
/// @return IoU for the couple
float computeIoU(const cv::Mat& result, const cv::Mat& pred);

/// @brief show on screen the results
/// @param source , path for the original images
/// @param path , path for the images
void showResults(const std::string& source, const std::string& path);


#endif // !EVAL_H
