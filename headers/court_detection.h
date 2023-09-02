#ifndef COURT_DETECTION_H
#define COURT_DETECTION_H
#include "header.h"

void player_elimination(cv::Mat image, cv::Mat& img_out, cv::Mat mask);

std::vector<double> color_quantization(cv::Mat image, cv::Mat& clustered, cv::Mat& centroid);

void field_distinction(cv::Mat image_box, cv::Mat clustered, cv::Mat& segmented_field);

void merge_clusters(cv::Mat& labels, cv::Mat& centers, float merge_threshold);

std::vector<double> maximum_distance(cv::Mat image, cv::Mat clustered_image, cv::Mat centers);

//actually not used 
void lines_detector(cv::Mat image);

//actually not used
void court_localization(cv::Mat image, cv::Mat& edges);

//actually not used 
void fill_image(cv::Mat& image);

/// @brief Finds the line that divides the playing field from the background
/// @param image where to find the line
/// @param longest_line the line that divides the playing field from the background if found
/// @return true if the line is found, false otherwise
bool line_refinement(cv::Mat& image, cv::Vec2f& longest_line);


/// @brief Adjusts the segmentation mask according to the line found
/// @param segmentation mask to be adjusted
/// @param line to use to adjust the mask
void court_segmentation_refinement(cv::Mat& segmentation, cv::Vec2f& line);

#endif // !COURT_DETECTION_H
