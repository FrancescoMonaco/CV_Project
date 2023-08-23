#ifndef UTILITY_H
#define UTILITY_H
// Utility.h : Francesco Pio Monaco
#include <opencv2/opencv.hpp>
#include "Eval.h"

/// @brief Checks if the image has a considerable amount of orange color
/// @param image , the image to be checked
/// @param orange_lower_bound , the lower bound for the orange color
/// @param orange_upper_bound , the upper bound for the orange color
/// @return true if the image has orange color, false otherwise
bool hasOrangeColor(const cv::Mat& image, int orange_lower_bound = 120, int orange_upper_bound = 255);

//***Rectangles Utility Functions***

/// @brief Removes the rectangles that have a uniform color from the vector
/// @param rects vector of rectangles
/// @param image where the color variation is calculated
/// @param threshold cutoff value for the color variation
void removeUniformRect(std::vector<cv::Rect>& rects, cv::Mat image, int threshold = 10);

/// @brief Removes the rectangles where one of the sides is too small
/// @param rects 
void removeFlatRect(std::vector<cv::Rect>& rects);

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

/// @brief Reshapes the vector of bounding boxes in a matrix of bounding boxes
/// @param bbs list of bounding boxes
/// @return vector for each image of the list of bounding boxes
std::vector<std::vector<cv::Rect>> reshapeBB(std::vector<BoundingBox> bbs, int NUM_IMAGES = 15);

//***Classification Functions***

/// @brief 
/// @param image the image where the rectangles are detected
/// @param rects the detected rectangles
/// @return a vector of labels for each rectangle
std::vector<int> classify(cv::Mat& image, std::vector<cv::Rect> rects);


#endif // !UTILITY_H