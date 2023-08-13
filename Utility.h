#ifndef UTILITY_H
#define UTILITY_H
#include <opencv2/opencv.hpp>
//Francesco Pio Monaco

/// @brief Checks if the image has a considerable amount of orange color
/// @param image , the image to be checked
/// @param orange_lower_bound , the lower bound for the orange color
/// @param orange_upper_bound , the upper bound for the orange color
/// @return true if the image has orange color, false otherwise
bool hasOrangeColor(const cv::Mat& image, int orange_lower_bound = 120, int orange_upper_bound = 255);

cv::Mat 
#endif // !UTILITY_H