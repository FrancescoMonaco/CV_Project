#include "../headers/player_detection.h"
#include <opencv2/photo.hpp>
#include <opencv2/photo/cuda.hpp>

// player_detection.cpp : Michele Russo

void player_segmentation(cv::Mat image, cv::Mat& seg_image, std::string str) {


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

			//cv::imshow("canny ", edges);
			//cv::waitKey(0);

			//close the edges found on canny 
			close_lines(edges);

			//cv::imshow("linee ", edges);
			//cv::waitKey(0);

			//add some lines at the boundaries of the edges
			create_lines(edges, edges);

			//i use this function to color inside the figures to get a first temporary segmentation
			fill_segments(edges);

			//cv::imshow("segmentation", edges);
			//cv::waitKey(0);

			//merging color clustering given at start and temporary segmentation computed just for the box
			super_impose(cluster, edges, parameters);

			//create the final mask for segmentation
			h = edges.rows;
			w = edges.cols;

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



			cv::destroyAllWindows();
		}


	}
	else {
		std::cout << "error path\n";
	}

}




void close_lines(cv::Mat& edge_image) {



	//give the size for the application of morphological operator  

	int morph_size = 3;

	cv::Mat element = getStructuringElement(cv::MORPH_CROSS
		, cv::Size(morph_size, morph_size));

	//perform gradient morphological 
	cv::Mat img_out;
	morphologyEx(edge_image, img_out, cv::MORPH_GRADIENT, element, cv::Point(-1, -1), 3);

	cv::Mat img_out1;
	morphologyEx(img_out, img_out1, cv::MORPH_ERODE, element, cv::Point(-1, -1), 1);

	edge_image = img_out1.clone();
}


void fill_segments(cv::Mat& edge_image) {


	std::vector<std::vector<cv::Point> > contours;
	std::vector<cv::Vec4i> hierarchy;
	cv::Mat dst = cv::Mat::zeros(edge_image.rows, edge_image.cols, CV_8UC3);

	// find contours of the image
	findContours(edge_image, contours, hierarchy,
		cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

	// iterate through all the top-level contours,
	// draw each connected component with its own random color
	int idx = 0;
	for (; idx >= 0; idx = hierarchy[idx][0])
	{
		cv::Scalar color(255, 255, 255);
		drawContours(edge_image, contours, idx, color, cv::FILLED, 4, hierarchy);
	}

	//cv::imshow("Components", edge_image);
	//cv::waitKey(0);
}




void clustering(cv::Mat image, cv::Mat& cluster) {


	int numClusters = 11; // Number of desired colors after quantization
	cv::Mat labels, centers;
	cv::TermCriteria criteria = cv::TermCriteria(cv::TermCriteria::EPS | cv::TermCriteria::MAX_ITER, 2, 0.01);

	//convert the image into HSV color space since it is more informative than RGB image
	cv::Mat image_box;
	cv::cvtColor(image, image_box, cv::COLOR_BGR2HSV);

	cv::Mat floatImage, clustered;

	std::vector<cv::Vec3b> vec;
	cv::Mat mask(image_box.rows, image_box.cols, CV_8UC1);
	//std::vector<cv::Point> pixel_positions;


	cv::Mat lbpImage(image.size(), CV_8U, cv::Scalar(0));

	//calculate local binary pattern to add more information about neighboorhood of the single pixel
	calculateLBP(image_box, lbpImage, 3, 5);

	for (int i = 0; i < image_box.rows; i++) {
		for (int j = 0; j < image_box.cols; j++) {

			vec.push_back(image_box.at<cv::Vec3b>(i, j));
			mask.at<uchar>(i, j) = 1;
			//pixel_positions.push_back(cv::Point(j, i)); // Store pixel positions


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

	//cv::Mat flat = image_box.reshape(1, image_box.cols * image_box.rows);
	cv::kmeans(flattened_data, numClusters, labels, criteria, 2, cv::KMEANS_PP_CENTERS, centers);

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


	//cv::imshow("clustering", cluster);
	//cv::waitKey(0);

}
void create_lines(cv::Mat edges, cv::Mat& output_edges) {

	// Vectors to store the starting and ending points of the lines
	std::vector<cv::Point> starters_up, starters_down;
	std::vector<cv::Point> terminators_up, terminators_down;


	bool start_up = false, start_down = false;

	starters_up.clear();
	terminators_up.clear();
	starters_down.clear();
	terminators_down.clear();

	int n_rows = edges.rows, col = edges.cols;

	//--------------------- UP-DOWN CLOSURE ------------------------------
	for (int j = 0; j < edges.cols; j++) { //in the same cycle we find the starting and ending point of the lines up and down
		// UP check
		uchar pixel_up = edges.at<uchar>(0, j);

		if (!start_up && pixel_up == 255 && j + 1 != edges.cols && edges.at<uchar>(0, j + 1) != 255) {

			starters_up.push_back(cv::Point(j, 0));
			start_up = true;
		}

		else if (start_up && pixel_up == 255) {

			start_up = false;
			terminators_up.push_back(cv::Point(j, 0));
		}
		// DOWN check
		uchar pixel_down = edges.at<uchar>(n_rows - 1, j);

		if (!start_down && pixel_down == 255 && j + 1 != edges.cols && edges.at<uchar>(n_rows - 1, j + 1) != 255) {

			starters_down.push_back(cv::Point(j, n_rows - 1));
			start_down = true;

		}

		else if (start_down && pixel_down == 255) {
			start_down = false;
			terminators_down.push_back(cv::Point(j, n_rows - 1));
		}

	}
	// UP closure
	if (start_up) {
		starters_up.pop_back();

	}

	for (int i = 0; i < starters_up.size(); i++) {

		//take starting and ending point
		cv::Point starte = starters_up[i];
		cv::Point end = terminators_up[i];

		if (end.x - starte.x > edges.rows / 4) {

		}
		else {
			for (int j = starte.x; j < end.x; j++) {

				edges.at<uchar>(0, j) = 255;
				edges.at<uchar>(1, j) = 255;
				edges.at<uchar>(2, j) = 255;
				edges.at<uchar>(3, j) = 255;
				edges.at<uchar>(4, j) = 255;

			}
		}

	}

	// DOWN closure
	if (start_down) {
		starters_down.pop_back();

	}


	for (int i = 0; i < starters_down.size(); i++) {

		//take starting and ending point
		cv::Point starte = starters_down[i];
		cv::Point end = terminators_down[i];

		if (end.x - starte.x > edges.rows / 2) {
			continue;
		}
		else {
			for (int j = starte.x; j < end.x; j++) {

				edges.at<uchar>(n_rows - 1, j) = 255;
				edges.at<uchar>(n_rows - 2, j) = 255;
				edges.at<uchar>(n_rows - 3, j) = 255;
				edges.at<uchar>(n_rows - 4, j) = 255;
				edges.at<uchar>(n_rows - 5, j) = 255;

			}
		}

	}
	//--------------------- LATERAL CLOSURE ------------------------------

	start_up = start_down = false;

	starters_up.clear();
	terminators_up.clear();
	starters_down.clear();
	terminators_down.clear();


	for (int j = 0; j < edges.rows; j++) { //we use the same cycle for left and right
		// LEFT check
		uchar pixel_up = edges.at<uchar>(j, 0);

		if (!start_up && pixel_up == 255 && j + 1 != edges.rows && edges.at<uchar>(j + 1, 0) != 255) {

			starters_up.push_back(cv::Point(0, j));
			start_up = true;

		}

		else if (start_up && pixel_up == 255) {

			start_up = false;
			terminators_up.push_back(cv::Point(0, j));
			//std::cout << "found\n";
		}

		// RIGHT check
		uchar pixel_down = edges.at<uchar>(j, col - 1);

		if (!start_down && pixel_down == 255 && j + 1 != edges.rows && edges.at<uchar>(j + 1, col - 1) != 255) {

			starters_down.push_back(cv::Point(col - 1, j));
			start_down = true;

		}

		else if (start_down && pixel_down == 255) {

			start_down = false;
			terminators_down.push_back(cv::Point(col - 1, j));
			//std::cout << "found\n";
		}

	}

	// LEFT closure
	if (start_up) {
		starters_up.pop_back();

	}

	for (int i = 0; i < starters_up.size(); i++) {


		cv::Point starte = starters_up[i];
		cv::Point end = terminators_up[i];

		if ((end.y - starte.y) > edges.rows / 2) {

		}
		else {
			for (int j = starte.y; j < end.y; j++) {

				edges.at<uchar>(j, 0) = 255;
				edges.at<uchar>(j, 1) = 255;
				edges.at<uchar>(j, 2) = 255;
				edges.at<uchar>(j, 3) = 255;
				edges.at<uchar>(j, 4) = 255;

			}
		}

	}

	// RIGHT closure
	if (start_down) {
		starters_down.pop_back();

	}


	for (int i = 0; i < starters_down.size(); i++) {


		cv::Point starte = starters_down[i];
		cv::Point end = terminators_down[i];

		if (end.y - starte.y > edges.cols / 3) {

		}
		else {
			for (int j = starte.y; j < end.y; j++) {

				edges.at<uchar>(j, col - 1) = 255;
				edges.at<uchar>(j, col - 2) = 255;
				edges.at<uchar>(j, col - 3) = 255;
				edges.at<uchar>(j, col - 4) = 255;
				edges.at<uchar>(j, col - 5) = 255;

			}
		}

	}

	output_edges = edges.clone();
	/*cv::imshow("new edges", edges);
	cv::waitKey(0);*/

}
bool sortbysec(const std::pair<int, cv::Vec3b>& a,
	const std::pair<int, cv::Vec3b>& b)
{
	return (a.first > b.first);
}

void super_impose(cv::Mat clustering, cv::Mat& mask, std::vector<int> box_parameters) {

	//take box parameters location
	int x = box_parameters[0];
	int y = box_parameters[1];
	int w = box_parameters[2];
	int h = box_parameters[3];

	//number of pixels detected as part of the temporary player 
	double n_nonzeros = cv::countNonZero(mask);

	//numbeer of pixels inside the image

	double tot = mask.cols * mask.rows;

	//number of black pixels
	double n_zeros = tot - n_nonzeros;

	//not used actually
	double more = 0.0;

	//box expansion when the area of the temporary segmentation is more than 75% than the total are of teh box
	if (n_nonzeros / tot > 0.64) {
		double num = n_nonzeros / tot;
		num = num * 20;
		if (n_nonzeros / tot > 90) {
			num = 30;

		}
		if (x + w + num <= clustering.cols) {

			w += num;

			cv::Mat paddedImage(mask.rows, mask.cols + static_cast<int>(num), mask.type(), cv::Vec3b(0, 0, 0));
			mask.copyTo(paddedImage(cv::Rect(0, 0, mask.cols, mask.rows)));
			mask = paddedImage.clone();
			//more = -0.1;
			n_zeros = n_zeros + (static_cast<int>(num) * mask.rows);
		}

		if (x - num >= 0) {
			x = x - num;
			cv::Mat paddedImage(mask.rows, mask.cols + static_cast<int>(num), mask.type(), cv::Vec3b(0, 0, 0));
			mask.copyTo(paddedImage(cv::Rect(static_cast<int>(num), 0, mask.cols, mask.rows)));
			mask = paddedImage.clone();
			//more = -0.1;
			w += num;
			n_zeros = n_zeros + (static_cast<int>(num) * mask.rows);
		}

	}

	cv::Mat box_superimpose(mask.size(), CV_8UC3);
	cv::Mat box(mask.size(), CV_8UC3);
	cv::Mat reverse_box(mask.size(), CV_8UC3);

	for (int i = y; i < y + h; i++) {
		for (int j = x; j < x + w; j++) {

			//super impose
			if (mask.at<uchar>(i - y, j - x) == 255) {

				box_superimpose.at<cv::Vec3b>(i - y, j - x) = clustering.at<cv::Vec3b>(i, j);
				reverse_box.at<cv::Vec3b>(i - y, j - x) = cv::Vec3b(0, 0, 0);
			}
			else {
				box_superimpose.at<cv::Vec3b>(i - y, j - x) = cv::Vec3b(0, 0, 0);
				reverse_box.at<cv::Vec3b>(i - y, j - x) = clustering.at<cv::Vec3b>(i, j);
			}

			box.at<cv::Vec3b>(i - y, j - x) = clustering.at<cv::Vec3b>(i, j);

		}

	}



	cv::Mat final_segmentation;
	std::vector<cv::Vec3b> colors;
	//std::vector<int> pixels;
	std::vector<std::pair<int, cv::Vec3b>> combinedVector;


	/*cv::imshow(" ", reverse_box);
	cv::waitKey(0);*/

	//find all the color outside the shape of the mask
	for (int i = 0; i < mask.rows; i++) {
		for (int j = 0; j < mask.cols; j++) {

			if (box_superimpose.at<cv::Vec3b>(i, j) == cv::Vec3b(0, 0, 0)) {

				cv::Vec3b color = box.at<cv::Vec3b>(i, j);

				auto it = std::find(colors.begin(), colors.end(), color);

				if (it != colors.end()) {
					continue;
				}
				else {

					cv::Mat temp;
					cv::inRange(reverse_box, color, color, temp);

					int pixel = cv::countNonZero(temp);

					combinedVector.push_back(std::pair(pixel, color));
					colors.push_back(color);
				}

			}

		}
	}

	//std::cout << "new image\n";
	double num_labels = colors.size();
	std::sort(combinedVector.begin(), combinedVector.end(), sortbysec);
	//i take only the pixel who are the most out 

	for (int z = 0; z < colors.size(); z++) {
		double n_elements = combinedVector[z].first;

		double fr = n_elements;

		double tr = n_zeros / num_labels;

		//skip
		if (fr <= tr) {
			continue;
		}
		else {
			cv::Vec3b color = combinedVector[z].second;
			for (int i = 0; i < mask.rows; i++) {
				for (int j = 0; j < mask.cols; j++) {


					if (box_superimpose.at<cv::Vec3b>(i, j) == color) {

						box_superimpose.at<cv::Vec3b>(i, j) = cv::Vec3b(0, 0, 0);
					}

				}
			}
			//cv::imshow(" ", box_superimpose);
			//cv::waitKey(0);
		}

	}

	/*cv::imshow("final", box_superimpose);
	cv::waitKey(0);
	*/
	cv::Mat final_seg, inversion;
	cv::inRange(box_superimpose, cv::Vec3b(0, 0, 0), cv::Vec3b(0, 0, 0), inversion);

	cv::bitwise_not(inversion, final_seg);


	//remove the non connected components
	remove_components(final_seg);

	//fix the box if it was exapanded 
	/*std::cout << box_parameters[0] <<std::endl;
	std::cout << box_parameters[2] <<std::endl;
	std::cout << x <<std::endl;*/


	cv::Rect roi(box_parameters[0] - x, 0, box_parameters[2] - 1, mask.rows);
	cv::Mat originalImage = final_seg(roi);
	//cv::imshow("final seg", originalImage);
	//cv::waitKey(0);

	mask = originalImage.clone();
}

void remove_components(cv::Mat& mask) {

	// Create a labeled image to store connected components
	cv::Mat labeledImage;
	int numLabels = cv::connectedComponents(mask, labeledImage, 8, CV_32S);

	// Create a vector to store the pixel counts for each region
	std::vector<int> regionPixelCounts(numLabels, 0);

	//count number of variables 
	double tot = 0;

	// Iterate through the labeled image and count pixels for each label
	for (int y = 0; y < labeledImage.rows; y++) {
		for (int x = 0; x < labeledImage.cols; x++) {
			int label = labeledImage.at<int>(y, x);

			if (label > 0) {
				tot++;
				regionPixelCounts[label]++;
			}

		}
	}


	double medium = tot / regionPixelCounts.size();

	//remove all the non connected componets
	for (int i = 0; i < mask.rows; i++) {
		for (int j = 0; j < mask.cols; j++) {
			int label = labeledImage.at<int>(i, j);
			//remove condition

			if (regionPixelCounts[label] < medium - 1) {
				//std::cout<< regionPixelCounts[label] <<std::endl;
				mask.at<uchar>(i, j) = 0;

			}
			else {
				continue;
			}

		}
	}

	//cv::imshow("final mask", mask);
	//cv::waitKey(0);
}



void calculateLBP(cv::Mat image, cv::Mat lbpImage, int radius, int neighbors) {
	cv::Mat grayImage;

	cv::cvtColor(image, grayImage, cv::COLOR_BGR2GRAY);


	for (int i = radius; i < grayImage.rows - radius; i++) {
		for (int j = radius; j < grayImage.cols - radius; j++) {
			uchar center = grayImage.at<uchar>(i, j);
			uchar code = 0;

			for (int k = 0; k < neighbors; k++) {
				int x = j + static_cast<int>(radius * cos(2.0 * CV_PI * k / neighbors));
				int y = i - static_cast<int>(radius * sin(2.0 * CV_PI * k / neighbors));

				if (grayImage.at<uchar>(y, x) >= center) {
					code |= (1 << k);
				}
			}

			lbpImage.at<uchar>(i, j) = code;
		}
	}


}

