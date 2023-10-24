#include "../headers/player_detection.h"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

// player_segmentation_robustness.cpp : Michele Russo

void player_segmentation_robust(cv::Mat image, cv::Mat& seg_image, std::string str) {


	std::ifstream file(str);
	cv::Mat mask = image.clone();


	cv::fastNlMeansDenoisingColored(image, image, 4.0, 10);

	cv::Mat cluster;

	//clusterize the image
	clustering(image, cluster, 11);

	if (file.is_open()) {

		std::string line;

		//take the string with the position of bounding box
		while (std::getline(file, line)) {
			std::vector<int> parameters;

			int x, y, w, h;

			std::istringstream iss(line);

			iss >> x >> y >> w >> h;

			cv::Mat img_out(h, w, CV_8UC3);

			//box position and dimension
			parameters.push_back(x);
			parameters.push_back(y);
			parameters.push_back(w);
			parameters.push_back(h);

			//isolate the box 
			cv::Mat mask_temp = mask.clone();
			for (int j = y; j < y + h; j++) {
				for (int i = x; i < x + w; i++) {

					img_out.at<cv::Vec3b>(j - y, i - x) = image.at<cv::Vec3b>(j, i);
					mask_temp.at<cv::Vec3b>(j, i) = image.at<cv::Vec3b>(j, i);

				}
			}

			//blur the image
			cv::Mat blur;
			cv::GaussianBlur(img_out, blur, cv::Size(5, 5), 0.8, 0.8);

			cv::Mat img_grey;
			cvtColor(blur, img_grey, cv::COLOR_BGR2GRAY);

			//start of computation gradient to create canny threshold 
			cv::Mat grad_x, grad_y;
			cv::Mat abs_grad_x, abs_grad_y, test_grad;

			cv::Sobel(img_grey, grad_x, CV_16S, 1, 0, 3, 1, 0, cv::BORDER_DEFAULT);
			cv::Sobel(img_grey, grad_y, CV_16S, 0, 1, 3, 1, 0, cv::BORDER_DEFAULT);
			cv::convertScaleAbs(grad_x, abs_grad_x);
			cv::convertScaleAbs(grad_y, abs_grad_y);
			cv::addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, test_grad);

			//Compute the median of the gradient magnitude
			cv::Scalar mean, stddev;
			cv::meanStdDev(test_grad, mean, stddev);
			double median = mean[0];
			int canny_c = 5;

			//calculaation of the gradient by using the threshold computed before 
			cv::Mat edges;
			cv::Canny(img_grey, edges, canny_c * median / 4, canny_c * median / 2);

			//cv::imshow("canny ", edges);
			//cv::waitKey(0);

			//close the edges found on canny 
			close_lines(edges);

			cv::RNG rng(12345);

			std::vector < std::vector<cv::Point> > contours;

			findContours(edges, contours, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);
			std::vector<std::vector<cv::Point> >hull(contours.size());

			cv::Mat m = cv::Mat::zeros(edges.size(), CV_8UC1);

			for (size_t i = 0; i < contours.size(); i++)
			{
				convexHull(contours[i], hull[i]);
				cv::Scalar color = cv::Scalar(255);  // Fill with white (255)
				cv::drawContours(m, hull, static_cast<int>(i), color, cv::FILLED);
			}

			//merging color clustering given at start and temporary segmentation computed just for the box
			super_impose(cluster, m, parameters);

			//create the final mask for segmentation
			h = m.rows;
			w = m.cols;

			for (int j = y; j < y + h; j++) {
				for (int i = x; i < x + w; i++) {

					//make the union of bodies in different boxes
					if (seg_image.at<uchar>(j, i) != 255) {
						seg_image.at<uchar>(j, i) = m.at<uchar>(j - y, i - x);
					}
					else {
						continue;
					}
				}
			}
		}

	}
	else {
		std::cout << "error path\n";
	}

}




void close_lines_robustness(cv::Mat& edge_image) {



	//give the size for the application of morphological operator  

	int morph_size = 3;

	cv::Mat element = getStructuringElement(cv::MORPH_ELLIPSE
		, cv::Size(morph_size, morph_size));

	//perform gradient morphological 
	cv::Mat img_out;
	morphologyEx(edge_image, img_out, cv::MORPH_GRADIENT, element, cv::Point(-1, -1), 1);

	//cv::Mat img_out1;
	//morphologyEx(img_out, img_out1, cv::MORPH_ERODE, element, cv::Point(-1, -1), 1);

	edge_image = img_out.clone();
}