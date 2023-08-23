#pragma once
#include"header.h"

void player_segmentation(cv::Mat image,cv::Mat seg_image,std::string str);

void close_lines(cv::Mat& edge_image);

void fill_segments(cv::Mat& edge_image);

void create_segmented_image(cv::Mat segmeted_filed,cv::Mat segmented_player,std::vector<int> coordinates  ,std::string save);

