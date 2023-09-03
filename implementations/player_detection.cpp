#include "../headers/player_detection.h"
#include <cmath>

void player_segmentation(cv::Mat image, cv::Mat& seg_image, std::string str) {


	std::ifstream file(str);
	cv::Mat mask = image.clone();
	//create_mask(image, mask, str);

	if (file.is_open()) {

		std::string line;





		//take the string with the position of bounding box
		while (std::getline(file, line)) {
			int x, y, w, h;

			std::istringstream iss(line);

			iss >> x >> y >> w >> h;

			cv::Mat img_out(h, w, CV_8UC3);
			//isolate the box CHECK

			cv::Mat mask_temp = mask.clone();
			for (int j = y; j < y + h; j++) {
				for (int i = x; i < x + w; i++) {

					img_out.at<cv::Vec3b>(j - y, i - x) = image.at<cv::Vec3b>(j, i);
					//mask_temp.at<cv::Vec3b>(j,i)= image.at<cv::Vec3b>(j, i);

				}
			}/*
			cv::imshow("box", img_out);
			cv::waitKey(0);
			*/
			//cv::Mat clustered;
			//clustering(mask_temp, clustered);

			cv::Mat blur;
			cv::GaussianBlur(img_out, blur, cv::Size(5, 5), 0.4, 0.4);

			cv::Mat img_grey;
			cvtColor(blur, img_grey, cv::COLOR_BGR2GRAY);

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

			//std::cout << "Median: " << median << std::endl;

			cv::Mat edges;


			cv::Canny(img_grey, edges, canny_c * median / 4, canny_c * median / 2);

			//		cv::imshow("canny ", edges);
				//	cv::waitKey(0);


					//close the lines found out by using the clustering and after removing the less important 
			close_lines(edges);

			//	cv::imshow(" ", edges);
				//cv::waitKey(0);

				//i use this function to color inside the figures
			fill_segments(edges);

			//cv::imshow(" ", edges);
			//cv::waitKey(0);


			//cv::imshow("segmentation", segmented_image);
			//cv::waitKey(0);

			cv::destroyAllWindows();


			cv::Mat img(edges.size(), CV_8UC3);

			std::vector<cv::Vec3b> colors;


			//create the final mask for segmentation

			for (int j = y; j < y + h; j++) {
				for (int i = x; i < x + w; i++) {

					//make the union of bodies in different boxes
					if (seg_image.at<uchar>(j, i) != 255) {
						seg_image.at<uchar>(j, i) = edges.at<uchar>(j - y, i - x);
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




void close_lines(cv::Mat& edge_image) {



	//give the size for the application of morphological operator  

	int morph_size = 5;

	cv::Mat element = getStructuringElement(cv::MORPH_ELLIPSE
		, cv::Size(morph_size, morph_size));

	cv::Mat img_out;
	morphologyEx(edge_image, img_out, cv::MORPH_GRADIENT, element, cv::Point(-1, -1), 2);
	//thin some edges

	cv::Mat img_out1;
	morphologyEx(img_out, img_out1, cv::MORPH_ERODE, element, cv::Point(-1, -1), 2);


	//cv::imshow("algo", img_out1);
	//cv::waitKey(0);





	edge_image = img_out1.clone();
}


void fill_segments(cv::Mat& edge_image) {

	std::vector<std::vector<cv::Point> > contours;
	std::vector<cv::Vec4i> hierarchy;
	cv::Mat dst = cv::Mat::zeros(edge_image.rows, edge_image.cols, CV_8UC3);

	findContours(edge_image, contours, hierarchy,
		cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

	// iterate through all the top-level contours,
	// draw each connected component with its own random color
	int idx = 0;
	for (; idx >= 0; idx = hierarchy[idx][0])
	{
		cv::Scalar color(255, 255, 255);
		drawContours(edge_image, contours, idx, color, cv::FILLED, 8, hierarchy);
	}

	//cv::imshow("Components", edge_image);
	//cv::waitKey(0);
}




void clustering(cv::Mat image_box, cv::Mat& cluster) {


	int numClusters = 8; // Number of desired colors after quantization
	cv::Mat labels, centers;
	cv::TermCriteria criteria = cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 10, 0.1);


	cv::Mat floatImage, clustered;

	std::vector<cv::Vec3b> vec;
	cv::Mat mask(image_box.rows, image_box.cols, CV_8UC1);
	std::vector<cv::Point> pixel_positions;

	for (int i = 0; i < image_box.rows; i++) {
		for (int j = 0; j < image_box.cols; j++) {

			if (image_box.at<cv::Vec3b>(i, j) == cv::Vec3b(0, 0, 0)) {

				mask.at<uchar>(i, j) = 0;

			}
			else {

				vec.push_back(image_box.at<cv::Vec3b>(i, j));
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
		//flattened_data.at<float>(i, 4) = pixel_positions[i].x;         // X
		//flattened_data.at<float>(i, 5) = pixel_positions[i].y;         // Y
	}



	//cv::Mat flat = image_box.reshape(1, image_box.cols * image_box.rows);
	cv::kmeans(flattened_data, numClusters, labels, criteria, 40, cv::KMEANS_PP_CENTERS, centers);

	// Define replacement colors
	cv::Vec3b colors[8];

	colors[0] = cv::Vec3b(255, 0, 0); // Red
	colors[1] = cv::Vec3b(0, 0, 255); // Blue
	colors[2] = cv::Vec3b(0, 255, 0); // green 
	colors[3] = cv::Vec3b(255, 255, 255);
	colors[4] = cv::Vec3b(255, 255, 0);
	colors[5] = cv::Vec3b(255, 0, 255);
	colors[6] = cv::Vec3b(0, 255, 255);
	colors[7] = cv::Vec3b(100, 100, 100);



	clustered = cv::Mat(image_box.rows, image_box.cols, CV_8UC3);


	int z = 0;

	for (int i = 0; i < image_box.rows; i++) {

		for (int j = 0; j < image_box.cols; j++) {

			if (mask.at<uchar>(i, j) == 1) {

				int el = labels.at<int>(0, z);
				clustered.at<cv::Vec3b>(i, j) = colors[el];
				z++;
			}
			else {
				clustered.at<cv::Vec3b>(i, j) = cv::Vec3b(0, 0, 0);

			}
		}

	}

	cluster = clustered.clone();


	//cv::imshow("clustering", cluster);
	//cv::waitKey(0);





}

void create_mask(cv::Mat image, cv::Mat& mask, std::string str) {

	std::ifstream file(str);


	if (file.is_open()) {

		std::string line;

		//take the string with the position of bounding box
		while (std::getline(file, line)) {
			int x, y, w, h;

			std::istringstream iss(line);

			iss >> x >> y >> w >> h;


			for (int j = y; j < y + h; j++) {
				for (int i = x; i < x + w; i++) {

					mask.at<cv::Vec3b>(j, i) = cv::Vec3b(0, 0, 0);

				}
			}

		}
	}
	else {
		std::cout << "error path\n";
	}

	cv::imshow("mask", mask);
	cv::waitKey(0);
}

