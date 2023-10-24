#ifndef COURT_DETECTION_H
#define COURT_DETECTION_H
#include "header.h"

/// @brief Creates an image where the players are removed
/// @param image original image 
/// @param img_out image where the segmented players are removed
/// @param mask with the segmented players
void player_elimination(cv::Mat image, cv::Mat& img_out, cv::Mat mask);

/// @brief Quantizes the image based on colors
/// @param image to quantize
/// @param clustered image that contains the quantization
/// @param centroid mat that contains the centroids of the clusters
void color_quantization(cv::Mat image, cv::Mat& clustered, cv::Mat& centroid);

/// @brief Creates an image where the field is segmented and the background is black
/// @param image_box original image
/// @param clustered clustering result
/// @param segmented_field processed image
void field_distinction(cv::Mat image_box, cv::Mat clustered, cv::Mat& segmented_field);

/// @brief Reduces the number of clusters by merging based on distance between centroids
/// @param labels segmentation map
/// @param centers centroids of the clusters
/// @param merge_threshold distance threshold to merge clusters
void merge_clusters(cv::Mat& labels, cv::Mat& centers, float merge_threshold);

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
