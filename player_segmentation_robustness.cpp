#include "player_detection.h"

#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
void player_segmentation_robust(cv::Mat image, cv::Mat& seg_image, std::string str) {


	std::ifstream file(str);
	cv::Mat mask = image.clone();


	cv::fastNlMeansDenoisingColored(image, image, 4.0, 10);

	cv::Mat cluster;

	//clusterize the image
	clustering(image, cluster);

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

			cv::imshow("canny ", edges);
			cv::waitKey(0);

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

			// Display 
			cv::imshow("Filled Convex Hull", m);
			
			cv::waitKey(0);

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



			cv::destroyAllWindows();
		}


	}
	else {
		std::cout << "error path\n";
	}

}




void close_lines_robustness(cv::Mat& edge_image) {



	//give the size for the application of morphological operator  

	int morph_size = 3;

	cv::Mat element = getStructuringElement(cv::MORPH_CROSS
		, cv::Size(morph_size, morph_size));

	//perform gradient morphological 
	cv::Mat img_out;
	morphologyEx(edge_image, img_out, cv::MORPH_GRADIENT, element, cv::Point(-1, -1), 1);

	//cv::Mat img_out1;
	//morphologyEx(img_out, img_out1, cv::MORPH_ERODE, element, cv::Point(-1, -1), 1);

	edge_image = img_out.clone();
}

//
//void fill_segments(cv::Mat& edge_image) {
//
//
//	std::vector<std::vector<cv::Point> > contours;
//	std::vector<cv::Vec4i> hierarchy;
//	cv::Mat dst = cv::Mat::zeros(edge_image.rows, edge_image.cols, CV_8UC3);
//
//	// find contours of the image
//	findContours(edge_image, contours, hierarchy,
//		cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
//
//	// iterate through all the top-level contours,
//	// draw each connected component with its own random color
//	int idx = 0;
//	for (; idx >= 0; idx = hierarchy[idx][0])
//	{
//		cv::Scalar color(255, 255, 255);
//		drawContours(edge_image, contours, idx, color, cv::FILLED, 4, hierarchy);
//	}
//
//	//cv::imshow("Components", edge_image);
//	//cv::waitKey(0);
//}


//
//void create_lines(cv::Mat edges, cv::Mat& output_edges) {
//
//	// Vectors to store the starting and ending points of the lines
//	std::vector<cv::Point> starters_up, starters_down;
//	std::vector<cv::Point> terminators_up, terminators_down;
//
//
//	bool start_up = false, start_down = false;
//
//	starters_up.clear();
//	terminators_up.clear();
//	starters_down.clear();
//	terminators_down.clear();
//
//	int n_rows = edges.rows, col = edges.cols;
//
//	//--------------------- UP-DOWN CLOSURE ------------------------------
//	for (int j = 0; j < edges.cols; j++) { //in the same cycle we find the starting and ending point of the lines up and down
//		// UP check
//		uchar pixel_up = edges.at<uchar>(0, j);
//
//		if (!start_up && pixel_up == 255 && j + 1 != edges.cols && edges.at<uchar>(0, j + 1) != 255) {
//
//			starters_up.push_back(cv::Point(j, 0));
//			start_up = true;
//		}
//
//		else if (start_up && pixel_up == 255) {
//
//			start_up = false;
//			terminators_up.push_back(cv::Point(j, 0));
//		}
//		// DOWN check
//		uchar pixel_down = edges.at<uchar>(n_rows - 1, j);
//
//		if (!start_down && pixel_down == 255 && j + 1 != edges.cols && edges.at<uchar>(n_rows - 1, j + 1) != 255) {
//
//			starters_down.push_back(cv::Point(j, n_rows - 1));
//			start_down = true;
//
//		}
//
//		else if (start_down && pixel_down == 255) {
//			start_down = false;
//			terminators_down.push_back(cv::Point(j, n_rows - 1));
//		}
//
//	}
//	// UP closure
//	if (start_up) {
//		starters_up.pop_back();
//
//	}
//
//	for (int i = 0; i < starters_up.size(); i++) {
//
//		//take starting and ending point
//		cv::Point starte = starters_up[i];
//		cv::Point end = terminators_up[i];
//
//		if (end.x - starte.x > edges.rows / 4) {
//
//		}
//		else {
//			for (int j = starte.x; j < end.x; j++) {
//
//				edges.at<uchar>(0, j) = 255;
//				edges.at<uchar>(1, j) = 255;
//				edges.at<uchar>(2, j) = 255;
//				edges.at<uchar>(3, j) = 255;
//				edges.at<uchar>(4, j) = 255;
//
//			}
//		}
//
//	}
//
//	// DOWN closure
//	if (start_down) {
//		starters_down.pop_back();
//
//	}
//
//
//	for (int i = 0; i < starters_down.size(); i++) {
//
//		//take starting and ending point
//		cv::Point starte = starters_down[i];
//		cv::Point end = terminators_down[i];
//
//		if (end.x - starte.x > edges.rows / 2) {
//			continue;
//		}
//		else {
//			for (int j = starte.x; j < end.x; j++) {
//
//				edges.at<uchar>(n_rows - 1, j) = 255;
//				edges.at<uchar>(n_rows - 2, j) = 255;
//				edges.at<uchar>(n_rows - 3, j) = 255;
//				edges.at<uchar>(n_rows - 4, j) = 255;
//				edges.at<uchar>(n_rows - 5, j) = 255;
//
//			}
//		}
//
//	}
//	//--------------------- LATERAL CLOSURE ------------------------------
//
//	start_up = start_down = false;
//
//	starters_up.clear();
//	terminators_up.clear();
//	starters_down.clear();
//	terminators_down.clear();
//
//
//	for (int j = 0; j < edges.rows; j++) { //we use the same cycle for left and right
//		// LEFT check
//		uchar pixel_up = edges.at<uchar>(j, 0);
//
//		if (!start_up && pixel_up == 255 && j + 1 != edges.rows && edges.at<uchar>(j + 1, 0) != 255) {
//
//			starters_up.push_back(cv::Point(0, j));
//			start_up = true;
//
//		}
//
//		else if (start_up && pixel_up == 255) {
//
//			start_up = false;
//			terminators_up.push_back(cv::Point(0, j));
//			//std::cout << "found\n";
//		}
//
//		// RIGHT check
//		uchar pixel_down = edges.at<uchar>(j, col - 1);
//
//		if (!start_down && pixel_down == 255 && j + 1 != edges.rows && edges.at<uchar>(j + 1, col - 1) != 255) {
//
//			starters_down.push_back(cv::Point(col - 1, j));
//			start_down = true;
//
//		}
//
//		else if (start_down && pixel_down == 255) {
//
//			start_down = false;
//			terminators_down.push_back(cv::Point(col - 1, j));
//			//std::cout << "found\n";
//		}
//
//	}
//
//	// LEFT closure
//	if (start_up) {
//		starters_up.pop_back();
//
//	}
//
//	for (int i = 0; i < starters_up.size(); i++) {
//
//
//		cv::Point starte = starters_up[i];
//		cv::Point end = terminators_up[i];
//
//		if ((end.y - starte.y) > edges.rows / 2) {
//
//		}
//		else {
//			for (int j = starte.y; j < end.y; j++) {
//
//				edges.at<uchar>(j, 0) = 255;
//				edges.at<uchar>(j, 1) = 255;
//				edges.at<uchar>(j, 2) = 255;
//				edges.at<uchar>(j, 3) = 255;
//				edges.at<uchar>(j, 4) = 255;
//
//			}
//		}
//
//	}
//
//	// RIGHT closure
//	if (start_down) {
//		starters_down.pop_back();
//
//	}
//
//
//	for (int i = 0; i < starters_down.size(); i++) {
//
//
//		cv::Point starte = starters_down[i];
//		cv::Point end = terminators_down[i];
//
//		if (end.y - starte.y > edges.cols / 3) {
//
//		}
//		else {
//			for (int j = starte.y; j < end.y; j++) {
//
//				edges.at<uchar>(j, col - 1) = 255;
//				edges.at<uchar>(j, col - 2) = 255;
//				edges.at<uchar>(j, col - 3) = 255;
//				edges.at<uchar>(j, col - 4) = 255;
//				edges.at<uchar>(j, col - 5) = 255;
//
//			}
//		}
//
//	}
//
//	output_edges = edges.clone();
//	/*cv::imshow("new edges", edges);
//	cv::waitKey(0);*/
//
//}

