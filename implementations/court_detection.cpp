#include"../headers/court_detection.h"
#include "../headers/segmentation.h"



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

void fill_image(cv::Mat& image) {
	//start filling the image


	for (int i = 0; i < image.rows; i++) {
		for (int j = 0; j < image.cols; j++) {
			//i fill the holes left by the remove of boxes
			if (image.at<cv::Vec3b>(i, j) == cv::Vec3b(0, 0, 0)) {
				cv::Vec3b color = image.at<cv::Vec3b>(i, j - 1);
				image.at<cv::Vec3b>(i, j) = color;
			}
		}
	}

	//visualize the image
	cv::imshow("image w/out boxes", image);
	cv::waitKey(0);

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


void color_quantization(cv::Mat image, cv::Mat& img_out) {




	int numClusters = 3; // Number of desired colors after quantization
	cv::Mat labels, centers;
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
		flattened_data.at<float>(i, 4) = pixel_positions[i].x;         // X
		flattened_data.at<float>(i, 5) = pixel_positions[i].y;         // Y
	}

	cv::normalize(flattened_data, flattened_data, 0, 1, cv::NORM_MINMAX);

	cv::Mat floatImage, clustered;
	image.convertTo(floatImage, CV_32FC3, 1.0 / 255.0);
	cv::Mat flat = floatImage.reshape(1, floatImage.rows * floatImage.cols);

	cv::kmeans(flattened_data, numClusters, labels, criteria, 100, cv::KMEANS_PP_CENTERS, centers);







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

				int el = labels.at<int>(0, z);
				clustered.at<cv::Vec3b>(i, j) = colors[el];
				z++;
			}
			else {
				clustered.at<cv::Vec3b>(i, j) = cv::Vec3b(0, 0, 0);

			}
		}

	}




	//lines_detector(clustered);

	//cv::Mat converted;
	//clustered.convertTo(converted, CV_8U);
	//clustered.convertTo(clustered, CV_8U);
	//cv::imshow("Original Image", image);
	cv::imshow("Quantized Image", clustered);
	cv::waitKey(0);
	img_out = clustered.clone();

	//lines_detector(img_out);


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
	cv::imshow("segmented field", segmented_field);
	cv::waitKey(0);
}


//localize the field lines by using hough transform not used 
void lines_detector(cv::Mat image) {


	cv::Mat edge;

	court_localization(image, edge);

	cv::Mat grey_edge;
	// Copy edges to the images that will display the results in BGR
	cvtColor(edge, grey_edge, cv::COLOR_GRAY2BGR);

	// Standard Hough Line Transform
	std::vector<cv::Vec2f> lines; // will hold the results of the detection
	HoughLines(edge, lines, 1, CV_PI / 50, 180, 50); // runs the actual detection

	// Draw the lines
	for (size_t i = 0; i < lines.size(); i++)
	{
		float rho = lines[i][0], theta = lines[i][1];
		cv::Point pt1, pt2;
		double a = cos(theta), b = sin(theta);
		double x0 = a * rho, y0 = b * rho;
		pt1.x = cvRound(x0 + 1000 * (-b));
		pt1.y = cvRound(y0 + 1000 * (a));
		pt2.x = cvRound(x0 - 1000 * (-b));
		pt2.y = cvRound(y0 - 1000 * (a));
		line(image, pt1, pt2, cv::Scalar(0, 0, 255), 3, cv::LINE_AA);
	}

	imshow("Detected Lines (in red) - Standard Hough Line Transform", image);
	cv::waitKey(0);
}


//localizza il contorno non usato
void court_localization(cv::Mat image, cv::Mat& edge) {


	//start by using canny to localize the lines 
	cv::Mat blur_img;

	cv::GaussianBlur(image, blur_img, cv::Size(9, 9), 0.5, 0.5);

	cv::Mat grad_x, grad_y;
	cv::Mat abs_grad_x, abs_grad_y, test_grad;

	cv::Sobel(blur_img, grad_x, CV_16S, 1, 0, 3, 1, 0, cv::BORDER_DEFAULT);
	cv::Sobel(blur_img, grad_y, CV_16S, 0, 1, 3, 1, 0, cv::BORDER_DEFAULT);
	cv::convertScaleAbs(grad_x, abs_grad_x);
	cv::convertScaleAbs(grad_y, abs_grad_y);
	cv::addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, test_grad);
	//Compute the median of the gradient magnitude
	cv::Scalar mean, stddev;
	cv::meanStdDev(test_grad, mean, stddev);
	double median = mean[0];
	int canny_c = 7;
	std::cout << "Median: " << median << std::endl;
	//cv::Mat edges;


	cv::Canny(image, edge, (canny_c * median) / 2, canny_c * median);

	cv::imshow("edges", edge);
	cv::waitKey(0);
}

bool line_refinement(cv::Mat& image, cv::Vec2f& longest_line)
{
	// Put the image in grayscale
	cv::Mat gray;
	cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);

	// Apply adaptive thresholding to create markers for watershed
	cv::Mat markers;
	cv::adaptiveThreshold(gray, markers, 255, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY_INV, 11, 8);
	cv::imshow("Markers", markers);
	// Noise reduction using morphological operations
	cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
	cv::Mat opening;
	cv::morphologyEx(markers, opening, cv::MORPH_OPEN, kernel, cv::Point(-1, -1), 2);

	// Sure background area
	cv::Mat sure_bg;
	cv::dilate(opening, sure_bg, kernel, cv::Point(-1, -1), 3);
	cv::imshow("Sure Background", sure_bg);
	// Finding sure foreground area using distance transform
	cv::Mat dist_transform;
	cv::distanceTransform(opening, dist_transform, cv::DIST_L2, 5);
	cv::normalize(dist_transform, dist_transform, 0, 1, cv::NORM_MINMAX);
	cv::Mat sure_fg;
	cv::threshold(dist_transform, sure_fg, 0.8, 1, cv::THRESH_BINARY);
	cv::imshow("Transform", dist_transform);
	// Finding unknown region
	cv::Mat sure_fg_8u;
	sure_fg.convertTo(sure_fg_8u, CV_8U);
	cv::Mat unknown = sure_bg - sure_fg_8u;

	// Marker labeling
	cv::Mat markers_cvt;
	sure_fg_8u.convertTo(markers_cvt, CV_32S);
	cv::connectedComponents(sure_fg_8u, markers_cvt);

	// Apply watershed algorithm
	cv::watershed(image, markers_cvt);

	// Assign different colors to different segmented regions
	cv::Mat segmented_image = cv::Mat::zeros(image.size(), CV_8UC3);
	for (int row = 0; row < markers_cvt.rows; ++row) {
		for (int col = 0; col < markers_cvt.cols; ++col) {
			int label = markers_cvt.at<int>(row, col);
			if (label == -1) {
				segmented_image.at<cv::Vec3b>(row, col) = cv::Vec3b(0, 0, 255); // Mark as red
			}
			else {
				segmented_image.at<cv::Vec3b>(row, col) = cv::Vec3b(0, 255, 0); // Mark with green for other regions
			}
		}
	}

	// Display the segmented image
	//cv::imshow("Segmented Image", segmented_image);
	//cv::waitKey(0);
	cv::Mat image_seg_gray;
	cv::cvtColor(segmented_image, image_seg_gray, cv::COLOR_BGR2GRAY);
	// Apply edge detection (e.g., Canny)
	cv::Mat edges;
	cv::Canny(image_seg_gray, edges, 50, 150);

	// Apply Hough Line Transform
	std::vector<cv::Vec2f> lines;
	cv::HoughLines(edges, lines, 1, CV_PI / 180, 100);

	// Find the longest horizontal line within the specified angle range
	float longest_line_length = 0;
	bool ret = false;
	for (const auto& line : lines) {
		float rho = line[0];
		float theta = line[1];

		// Convert theta to degrees
		float angle_deg = theta * 180.0f / CV_PI;

		if (std::abs(angle_deg - 90) <= 15 || std::abs(angle_deg - 270) <= 15) {
			// Check if the line is not too close to the image borders
			if (std::abs(rho) > 70 && std::abs(rho) < std::min(image_seg_gray.rows, image_seg_gray.cols) - 70) {
				// Check if the line is not close to the top or bottom of the image
				if (std::abs(rho) > 70 && std::abs(rho) < std::max(image_seg_gray.rows, image_seg_gray.cols) - 70) {
					float length = std::abs(rho);
					if (length > longest_line_length) {
						ret = true;
						longest_line_length = length;
						longest_line = line;
					}
				}
			}
		}
	}

	// Compute the endpoints
	double a = std::cos(longest_line[1]);
	double b = std::sin(longest_line[1]);
	double x0 = a * longest_line[0];
	double y0 = b * longest_line[0];
	cv::Point pt1(cvRound(x0 + 1000 * (-b)), cvRound(y0 + 1000 * (a)));
	cv::Point pt2(cvRound(x0 - 1000 * (-b)), cvRound(y0 - 1000 * (a)));


	//***JUST FOR DEGUB REMOVE AFTER***
	// Draw the longest horizontal line on the image
	cv::Mat image_with_line;
	cv::cvtColor(edges, image_with_line, cv::COLOR_GRAY2BGR);
	cv::line(image_with_line, pt1, pt2, cv::Scalar(0, 0, 255), 3);

	// Display the image with the longest horizontal line
	cv::imshow("Longest Horizontal Line", image_with_line);
	cv::waitKey(0);

	// Set the condition of usage
	return ret;
}
