#ifndef PLAYER_DETECTION_H
#define PLAYER_DETECTION_H
#include"header.h"

//***Classic segmentations definitions***//

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
/// @param num_clusters number of clusters
void clustering(cv::Mat image_box, cv::Mat& clustered, int num_clusters);

/// @brief Closes the edges of an edge image to form complete lines
/// @param edges edge image
/// @param output_edges closed edges
void create_lines(cv::Mat edges, cv::Mat& output_edges);

/// @brief temporarily expand the box for a better clustering and remove the small connected components
/// @param clustering input image
/// @param mask output image
/// @param box_parameters parameters of the box
void super_impose(cv::Mat clustering, cv::Mat& mask, std::vector<int> box_parameters);

/// @brief Compares two pairs of int and Vec3b
/// @param a pair of int and Vec3b
/// @param b pair of int and Vec3b
/// @return true if a is more than b
bool sortbysec(const std::pair<int, cv::Vec3b>& a,
	const std::pair<int, cv::Vec3b>& b);

/// @brief Removes all the connected components whose dimension is less than the biggest one
void remove_components(cv::Mat& mask);

/// @brief Computes the LBP of the image
/// @param image input image
/// @param lbpImage output image
/// @param radius used for the LBP
/// @param neighbors used to compute the LBP
void calculateLBP(cv::Mat image, cv::Mat lbpImage, int radius, int neighbors);

//***Robustness segmentations definitions***//

/// @brief Segment the players from the image using the convex hull
/// @param image original image
/// @param seg_image segmented image with the players
/// @param str stings where the bounding box is stored
void player_segmentation_robust(cv::Mat image, cv::Mat& seg_image, std::string str);

/// @brief Line closure using Convex Hull
/// @param edge_image Canny edge image
void close_lines_robustness(cv::Mat& edge_image);

#endif // !PLAYER_DETECTION_H