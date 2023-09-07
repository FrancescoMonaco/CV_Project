#ifndef UTILITY_H
#define UTILITY_H
// Utility.h : Francesco Pio Monaco
#include <opencv2/opencv.hpp>
#include "Eval.h"

//***Rectangles Utility Functions***

/// @brief Removes the rectangles that have a uniform color from the vector
/// @param rects vector of rectangles
/// @param image where the color variation is calculated
/// @param threshold cutoff value for the color variation
void removeUniformRect(std::vector<cv::Rect>& rects, cv::Mat image, int threshold = 10);

/// @brief Resizes the rectangle to the size of the image
void resizeRect(cv::Rect& r, cv::Mat image);

/// @brief Merges the rectangles that overlap of the percentage specified by the threshold
/// @param rects vector of rectangles
/// @param threshold value of the percentage of overlap
void mergeOverlapRect(std::vector<cv::Rect>& rects, int threshold = 10);

/// @brief Takes the raw rectangles and cleans them by removing the uniform ones and merging the overlapping ones
/// @param rects vector of rectangles
/// @param image where the rectangles are detected
void cleanRectangles(std::vector<cv::Rect>& rects, cv::Mat image);

//***Other Utility Functions***

/// @brief Computes the heat diffusion of the image using a heat equation
/// @param image 
/// @return the original image masked with the heat diffusion
cv::Mat computeDiffusion(cv::Mat image);

/// @brief Reshapes the vector of bounding boxes in a matrix of Rects
/// @param bbs list of bounding boxes
/// @return vector for each image of the list of Rects
std::vector<std::vector<cv::Rect>> reshapeBB(std::vector<BoundingBox> bbs, int NUM_IMAGES = 15);

//***Classification Functions***

/// @brief 
/// @param image the image where the rectangles are detected
/// @param rects the detected rectangles
/// @return a vector of labels for each rectangle
std::vector<int> classify(cv::Mat& image, std::vector<cv::Rect> rects);

/// @brief Writes the bounding boxes on a file
/// @param image the image where the rectangles are detected
/// @param rects the detected rectangles
/// @param labels the labels of the rectangles
/// @param rel_path the relative path of the file
void writeBB(cv::Mat& image, std::vector<cv::Rect> rects, std::vector<int> labels, std::string rel_path);

//***Segmentation Functions***

/// @brief Unites the playing field segmentation with the segmentation of the players
/// @param segmentation playing field segmentation
/// @param players mask with the player segmentation
/// @param rects rects of the players
/// @param labels labels of the players
void unifySegmentation(cv::Mat& segmentation, cv::Mat& players, std::vector<cv::Rect> rects, std::vector<int> labels);

/// @brief Transforms the segmentation matrix into a gray scale image
/// @param segmentation original segmentation matrix
/// @param gray_seg segmentation matrix converted to gray scale
void createSegmentationPNG(cv::Mat segmentation, cv::Mat& gray_seg);

/// @brief Writes the segmentation matrix on a file
/// @param image segmentation matrix
/// @param rel_path path of the file
void writeSEG(cv::Mat& image, std::string rel_path);


#endif // !UTILITY_H