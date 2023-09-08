#pragma once
#include"header.h"

void player_segmentation(cv::Mat image,cv::Mat& seg_image,std::string str);

void close_lines(cv::Mat& edge_image);

void fill_segments(cv::Mat& edge_image);

void clustering(cv::Mat image_box ,cv::Mat& clustered);

void create_mask(cv::Mat image, cv::Mat& mask, std::string str);

void create_lines(cv::Mat edges, cv::Mat& output_edges);

void super_impose(cv::Mat clustering, cv::Mat& mask, std::vector<int> box_parameters);

bool sortbysec(const std::pair<int, cv::Vec3b>& a,
	const std::pair<int, cv::Vec3b>& b);

void remove_components(cv::Mat& mask);

