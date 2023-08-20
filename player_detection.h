#pragma once
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <string>
#include <iostream>
#include <vector>

void player_segmentation(cv::Mat image,cv::Mat seg_image,std::string str);


void heat_diffusion(cv::Mat image);