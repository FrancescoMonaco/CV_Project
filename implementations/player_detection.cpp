#include "../headers/player_detection.h"
void player_segmentation(cv::Mat image, cv::Mat& seg_image, std::string str) {


	std::ifstream file(str);


	if (file.is_open()) {

		std::string line;

		//take the string with the position of bounding box
		while (std::getline(file, line)) {
			int x, y, w, h;

			std::istringstream iss(line);

			iss >> x >> y >> w >> h;

			cv::Mat img_out(h, w, CV_8UC3);
			//isolate the box CHECK

			for (int j = y; j < y + h; j++) {
				for (int i = x; i < x + w; i++) {

					img_out.at<cv::Vec3b>(j - y, i - x) = image.at<cv::Vec3b>(j, i);
				}
			}/*
			cv::imshow("box", img_out);
			cv::waitKey(0);
			*/
			cv::Mat clustered;
			clustering(img_out, clustered);


			cv::Mat img_grey;
			cvtColor(clustered, img_grey, cv::COLOR_BGR2GRAY);

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


			cv::Canny(img_grey, edges, canny_c * median / 4, canny_c * median / 2, 3, true);

			// Find contours in the edge image
			std::vector<std::vector<cv::Point>> contours;
			cv::findContours(edges, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

			// Define a threshold for edge length
			double minLength = 50.0;

			// Create a mask to keep edges longer than the threshold
			cv::Mat mask = cv::Mat::zeros(edges.size(), CV_8U);

			for (size_t i = 0; i < contours.size(); i++) {

				if (cv::arcLength(contours[i], true) >= minLength) {
					std::vector<std::vector<cv::Point>> contourSubset(1, contours[i]);
					cv::drawContours(mask, contourSubset, -1, cv::Scalar(255), cv::LINE_8);

				}
			}


			cv::Mat displayMask;
			mask.convertTo(displayMask, CV_8UC1);

			cv::imshow("Mask Image", displayMask);
			cv::waitKey(0);

			//close the lines found out by using the clustering and after removing the less important 
			close_lines(displayMask);

			/*cv::imshow(" ", displayMask);
			cv::waitKey(0);*/

			//i use this function to color inside the figures
			fill_segments(displayMask);

			/*cv::imshow(" ", displayMask);
			cv::waitKey(0);
			*/
			//errore
			//cv::imshow("segmentation", segmented_image);
			//cv::waitKey(0);

			cv::destroyAllWindows();


			//create the final mask for segmentation

			for (int j = y; j < y + h; j++) {
				for (int i = x; i < x + w; i++) {

					seg_image.at<uchar>(j, i) = displayMask.at<uchar>(j - y, i - x);
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
	morphologyEx(edge_image, img_out, cv::MORPH_DILATE, element, cv::Point(-1, -1), 3);
	//thin some edges
	//cv::Mat img_out1;
	//morphologyEx(img_out, img_out1, cv::MORPH_ERODE, element, cv::Point(-1,-1), 1);


	/*cv::imshow("algo", img_out1);
	cv::waitKey(0);*/





	edge_image = img_out.clone();
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


	int numClusters = 5; // Number of desired colors after quantization
	cv::Mat labels, centers;
	cv::TermCriteria criteria = cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 10, 0.1);


	cv::Mat floatImage, clustered;
	image_box.convertTo(floatImage, CV_32FC3, 1.0 / 255.0);
	cv::Mat flat = floatImage.reshape(1, floatImage.rows * floatImage.cols);
	cv::normalize(flat, flat, 0, 1, cv::NORM_MINMAX);

	//cv::Mat flat = image_box.reshape(1, image_box.cols * image_box.rows);
	cv::kmeans(flat, numClusters, labels, criteria, 150, cv::KMEANS_PP_CENTERS, centers);

	// Define replacement colors
	cv::Vec3b colors[5];

	colors[0] = cv::Vec3b(255, 0, 0); // Red
	colors[1] = cv::Vec3b(0, 0, 255); // Blue
	colors[2] = cv::Vec3b(0, 255, 0); // green 
	colors[3] = cv::Vec3b(255, 255, 255);
	colors[4] = cv::Vec3b(255, 255, 0);

	clustered = cv::Mat(image_box.rows, image_box.cols, CV_8UC3);


	for (int i = 0; i < image_box.rows * image_box.cols; i++) {

		int el = static_cast<int>(labels.at<int>(i));
		clustered.at<cv::Vec3b>(i / image_box.cols, i % image_box.cols) = colors[el];


	}


	//
	/*cv::imshow("clustered", clustered);

	cv::waitKey(0);
	*/
	cluster = clustered.clone();

}

