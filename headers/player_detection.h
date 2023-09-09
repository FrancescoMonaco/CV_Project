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

/// @brief Creates a mask on the image using the bounding box stored in the file
/// @param image input image
/// @param mask output mask
/// @param str path where the bounding box is stored
void create_mask(cv::Mat image, cv::Mat& mask, std::string str);

/// @brief Closes the edges of an edge image to form complete lines
/// @param edges edge image
/// @param output_edges closed edges
void create_lines(cv::Mat edges, cv::Mat& output_edges);

void super_impose(cv::Mat clustering, cv::Mat& mask, std::vector<int> box_parameters);

/// @brief Compares two pairs of int and Vec3b
/// @param a pair of int and Vec3b
/// @param b pair of int and Vec3b
/// @return true if a is more than b
bool sortbysec(const std::pair<int, cv::Vec3b>& a,
	const std::pair<int, cv::Vec3b>& b);

void remove_components(cv::Mat& mask);

#endif // !PLAYER_DETECTION_H