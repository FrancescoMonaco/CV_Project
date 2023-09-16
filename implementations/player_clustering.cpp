#include "../headers/player_detection.h"

// player_clustering.cpp : Michele Russo

void clustering(cv::Mat image, cv::Mat& cluster) {


	int numClusters = 11; // Number of desired colors after quantization
	cv::Mat labels, centers;
	cv::TermCriteria criteria = cv::TermCriteria(cv::TermCriteria::EPS | cv::TermCriteria::MAX_ITER, 200, 0.001);

	//convert the image into HSV color space since it is more informative than RGB image
	cv::Mat image_box;
	cv::cvtColor(image, image_box, cv::COLOR_BGR2HSV);

	// Data for the processing
	cv::Mat floatImage, clustered;

	std::vector<cv::Vec3b> vec;
	cv::Mat mask(image_box.rows, image_box.cols, CV_8UC1);


	cv::Mat lbpImage(image.size(), CV_8U, cv::Scalar(0));

	//calculate local binary pattern to add more information about neighboorhood of the single pixel
	calculateLBP(image_box, lbpImage, 3, 5);

	for (int i = 0; i < image_box.rows; i++) {
		for (int j = 0; j < image_box.cols; j++) {

			vec.push_back(image_box.at<cv::Vec3b>(i, j));
			mask.at<uchar>(i, j) = 1;
		}
	}

	// Convert Vec3b data to a format suitable for K-means
	cv::Mat flattened_data(vec.size(), 4, CV_32F);

	for (int i = 0; i < vec.size(); i++) {

		flattened_data.at<float>(i, 0) = vec[i][0];
		flattened_data.at<float>(i, 1) = vec[i][1];
		flattened_data.at<float>(i, 2) = vec[i][2];
		flattened_data.at<float>(i, 3) = lbpImage.at<uchar>(i / lbpImage.cols, i % lbpImage.cols);

	}

	//normalize data
	cv::normalize(flattened_data, flattened_data, 0, 1, cv::NORM_MINMAX);

	cv::kmeans(flattened_data, numClusters, labels, criteria, 5, cv::KMEANS_PP_CENTERS, centers);

	// Define replacement colors
	cv::Vec3b colors[15];

	colors[0] = cv::Vec3b(255, 0, 0);
	colors[1] = cv::Vec3b(0, 0, 255);
	colors[2] = cv::Vec3b(0, 255, 0);
	colors[3] = cv::Vec3b(255, 255, 255);
	colors[4] = cv::Vec3b(255, 255, 0);
	colors[5] = cv::Vec3b(255, 0, 255);
	colors[6] = cv::Vec3b(0, 255, 255);
	colors[7] = cv::Vec3b(100, 100, 100);
	colors[8] = cv::Vec3b(0, 100, 100);
	colors[9] = cv::Vec3b(100, 0, 100);
	colors[10] = cv::Vec3b(100, 100, 0);
	colors[11] = cv::Vec3b(150, 150, 150);
	colors[12] = cv::Vec3b(200, 200, 200);
	colors[13] = cv::Vec3b(50, 50, 50);
	colors[14] = cv::Vec3b(150, 200, 50);



	clustered = cv::Mat(image_box.rows, image_box.cols, CV_8UC3);

	//add colors on the image
	int z = 0;

	for (int i = 0; i < image_box.rows; i++) {

		for (int j = 0; j < image_box.cols; j++) {

			if (mask.at<uchar>(i, j) == 1) {

				int el = labels.at<int>(z);
				clustered.at<cv::Vec3b>(i, j) = colors[el];
				z++;
			}
			else {
				clustered.at<cv::Vec3b>(i, j) = cv::Vec3b(0, 0, 0);

			}
		}

	}

	cluster = clustered.clone();
}


bool sortbysec(const std::pair<int, cv::Vec3b>& a,
	const std::pair<int, cv::Vec3b>& b)
{
	return (a.first > b.first);
}


void calculateLBP(cv::Mat image, cv::Mat lbpImage, int radius, int neighbors) {
	cv::Mat grayImage;

	//convert the image into a grayscale image
	cv::cvtColor(image, grayImage, cv::COLOR_BGR2GRAY);

	//convert the image into a circle
	for (int i = radius; i < grayImage.rows - radius; i++) {
		for (int j = radius; j < grayImage.cols - radius; j++) {
			uchar center = grayImage.at<uchar>(i, j);
			uchar code = 0;

			//for each central pixel we look at its neighbourhood
			for (int k = 0; k < neighbors; k++) {
				//calculate the pixel location into the neigbourhood

				int x = j + static_cast<int>(radius * cos(2.0 * CV_PI * k / neighbors));
				int y = i - static_cast<int>(radius * sin(2.0 * CV_PI * k / neighbors));

				//for all the pixels that have a value greater than the central pixel, we 
				//perform the bitwise or and a shift operation  

				if (grayImage.at<uchar>(y, x) >= center) {
					code |= (1 << k);
				}
			}
			//final value of the pixel 
			lbpImage.at<uchar>(i, j) = code;
		}
	}
}