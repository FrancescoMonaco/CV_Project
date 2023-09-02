#ifndef PLAYER_DETECTION_H
#define PLAYER_DETECTION_H
#include"header.h"

void player_segmentation(cv::Mat image, cv::Mat& seg_image, std::string str);

void close_lines(cv::Mat& edge_image);

void fill_segments(cv::Mat& edge_image);

void clustering(cv::Mat image_box, cv::Mat& clustered);

void create_mask(cv::Mat image, cv::Mat& mask, std::string str);


#endif // !PLAYER_DETECTION_H