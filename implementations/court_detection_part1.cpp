#include"../headers/court_detection.h"
#include "../headers/segmentation.h"

// court_detection_part1.cpp : Michele Russo

//i don't consider the players which have their own segmentation path
void player_elimination(cv::Mat image, cv::Mat& img_out, cv::Mat mask)
{
	//clone the original image
	cv::Mat usage = image.clone();

	for (int i = 0; i < mask.rows; i++) {
		for (int j = 0; j < mask.cols; j++) {
			if (mask.at<uchar>(i, j) == 255) {
				//create a mask whithout considering the players, that will be segmented apart
				usage.at<cv::Vec3b>(i, j) = cv::Vec3b(0, 0, 0);
			}
		}
	}
	img_out = usage.clone();

}

void merge_clusters(cv::Mat& labels, cv::Mat& centers, float merge_threshold) {
	std::map<int, int> cluster_map;
	int num_clusters = centers.rows;

	for (int i = 0; i < num_clusters; ++i) {
		if (cluster_map.find(i) == cluster_map.end()) {
			for (int j = i + 1; j < num_clusters; ++j) {
				if (cluster_map.find(j) == cluster_map.end()) {
					float distance = cv::norm(centers.row(i), centers.row(j), cv::NORM_L2);
					if (distance < merge_threshold) {
						cluster_map[j] = i;
					}
				}
			}
		}
	}

	for (int i = 0; i < labels.rows; ++i) {
		int label = labels.at<int>(i);
		if (cluster_map.find(label) != cluster_map.end()) {
			labels.at<int>(i) = cluster_map[label];
		}
	}
}


void color_quantization(cv::Mat image, cv::Mat& img_out, cv::Mat& centers) {

	int numClusters = 3; // Number of desired colors after quantization
	cv::Mat labels;
	cv::TermCriteria criteria = cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 10, 0.1);

	std::vector<cv::Vec3b> vec;
	cv::Mat mask(image.rows, image.cols, CV_8UC1);
	std::vector<cv::Point> pixel_positions;

	for (int i = 0; i < image.rows; i++) {
		for (int j = 0; j < image.cols; j++) {

			if (image.at<cv::Vec3b>(i, j) == cv::Vec3b(0, 0, 0)) {

				mask.at<uchar>(i, j) = 0;

			}
			else {

				vec.push_back(image.at<cv::Vec3b>(i, j));
				mask.at<uchar>(i, j) = 1;
				pixel_positions.push_back(cv::Point(j, i)); // Store pixel positions

			}
		}
	}



	// Convert Vec3b data to a format suitable for K-means
	cv::Mat flattened_data(vec.size(), 3, CV_32F);

	for (size_t i = 0; i < vec.size(); ++i) {
		flattened_data.at<float>(i, 0) = vec[i][0];
		flattened_data.at<float>(i, 1) = vec[i][1];
		flattened_data.at<float>(i, 2) = vec[i][2];
	}

	cv::normalize(flattened_data, flattened_data, 0, 1, cv::NORM_MINMAX);

	cv::Mat floatImage, clustered;
	image.convertTo(floatImage, CV_32FC3, 1.0 / 255.0);
	cv::Mat flat = floatImage.reshape(1, floatImage.rows * floatImage.cols);

	cv::kmeans(flattened_data, numClusters, labels, criteria, 10, cv::KMEANS_PP_CENTERS, centers);


	float merge_threshold = 0.35;
	merge_clusters(labels, centers, merge_threshold);

	// Define replacement colors
	cv::Vec3b colors[3];

	colors[0] = cv::Vec3b(255, 0, 0); // Red
	colors[1] = cv::Vec3b(0, 0, 255); // Blue
	colors[2] = cv::Vec3b(0, 255, 0); // green 

	clustered = cv::Mat(image.rows, image.cols, CV_8UC3);

	int z = 0;

	for (int i = 0; i < image.rows; i++) {

		for (int j = 0; j < image.cols; j++) {

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

	img_out = clustered.clone();

	return;
}


void field_distinction(cv::Mat image_box, cv::Mat clustered, cv::Mat& segmented_field) {

	//colour used to distict field and background
	cv::Vec3b green(0, 255, 0);

	cv::Mat mask1, mask2, mask3;
	cv::inRange(clustered, cv::Vec3b(255, 0, 0), cv::Vec3b(255, 0, 0), mask1);

	cv::inRange(clustered, cv::Vec3b(0, 0, 255), cv::Vec3b(0, 0, 255), mask2);

	cv::inRange(clustered, cv::Vec3b(0, 255, 0), cv::Vec3b(0, 255, 0), mask3);


	int pixel_count1 = cv::countNonZero(mask1);
	int pixel_count2 = cv::countNonZero(mask2);
	int pixel_count3 = cv::countNonZero(mask3);

	int larger = std::max(pixel_count1, pixel_count2);
	larger = std::max(larger, pixel_count3);


	if (pixel_count1 == larger) {
		std::cout << "1\n";

		segmented_field.setTo(green, mask1);
		segmented_field.setTo(cv::Vec3b(0, 0, 0), mask2);
		segmented_field.setTo(cv::Vec3b(0, 0, 0), mask3);

	}
	else if (pixel_count2 == larger) {
		std::cout << "2\n";

		segmented_field.setTo(green, mask2);
		segmented_field.setTo(cv::Vec3b(0, 0, 0), mask1);
		segmented_field.setTo(cv::Vec3b(0, 0, 0), mask3);

	}
	else {
		std::cout << "3\n";


		segmented_field.setTo(green, mask3);
		segmented_field.setTo(cv::Vec3b(0, 0, 0), mask2);
		segmented_field.setTo(cv::Vec3b(0, 0, 0), mask1);
	}
}