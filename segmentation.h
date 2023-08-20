#pragma once
#include <iostream>
#include <omp.h>
//#include <opencv/cv.hpp>
#include "opencv2/highgui/highgui.hpp"
#include <vector>


struct pixel {
    int x;
    int y;
};


struct edge {
    float weigh;
    pixel node1;
    pixel node2;
};




int segmentation(cv::Mat img1, cv::Mat &img_out);

int root(std::vector <int>& Arr, int i);

float weight(int p1[], int p2[]);


