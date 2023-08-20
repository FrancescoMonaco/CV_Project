#pragma once
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <string>
#include <iostream>
#include <vector>

void box_elimination(cv::Mat image,cv::Mat img_out,std::string str );

void fill_image(cv::Mat image);

void court_localization(cv::Mat image);

void color_quantization(cv::Mat image);