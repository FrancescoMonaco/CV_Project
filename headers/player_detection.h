#ifndef PLAYER_DETECTION_H
#define PLAYER_DETECTION_H
#include"header.h"

/// @brief Segment the players from the image
/// @param image original image
/// @param seg_image image that stores the segmented players
/// @param str path where the bounding box is stored
void player_segmentation(cv::Mat image, cv::Mat& seg_image, std::string str);

/// @brief Applies morphological operations to the image
/// @param edge_image Canny edge image
void close_lines(cv::Mat& edge_image);

/// @brief Fills the holes in the image
/// @param edge_image edge image
void fill_segments(cv::Mat& edge_image);

/// @brief Clusters the image using the k-means algorithm
/// @param image_box input image
/// @param clustered output array
void clustering(cv::Mat image_box, cv::Mat& clustered);

void create_mask(cv::Mat image, cv::Mat& mask, std::string str);

void create_lines(cv::Mat edges, cv::Mat& output_edges);

void super_impose(cv::Mat clustering, cv::Mat& mask, std::vector<int> box_parameters);

bool sortbysec(const std::pair<int, cv::Vec3b>& a,
	const std::pair<int, cv::Vec3b>& b);


#endif // !PLAYER_DETECTION_H