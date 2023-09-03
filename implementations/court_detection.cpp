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

	//std::vector<double> distances;
	//distances = maximum_distance(image, clustered, centers);


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



bool line_refinement(cv::Mat& image, cv::Vec2f& longest_line) {
	// Put the image in grayscale
	cv::Mat gray;
	cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);

	// Apply adaptive thresholding to create markers for watershed
	cv::Mat markers;
	cv::adaptiveThreshold(gray, markers, 255, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY_INV, 11, 7);

	// Noise reduction using morphological operations
	cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
	cv::Mat opening;
	cv::morphologyEx(markers, opening, cv::MORPH_OPEN, kernel, cv::Point(-1, -1), 2);

	// Sure background area
	cv::Mat sure_bg;
	cv::dilate(opening, sure_bg, kernel, cv::Point(-1, -1), 3);

	// Finding sure foreground area using distance transform
	cv::Mat dist_transform;
	cv::distanceTransform(opening, dist_transform, cv::DIST_L2, 5);
	cv::normalize(dist_transform, dist_transform, 0, 1, cv::NORM_MINMAX);
	cv::Mat sure_fg;
	cv::threshold(dist_transform, sure_fg, 0.79, 1, cv::THRESH_BINARY);

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
	int middle_row = image_seg_gray.rows / 2;

	// Find the longest horizontal line within the specified angle range
	bool ret = false;
	for (const auto& line : lines) {
		float rho = line[0];
		float theta = line[1];

		// Convert theta to degrees
		float angle_deg = theta * 180.0f / CV_PI;

		// Check if the line is almost horizontal
		if (std::abs(angle_deg - 90) <= 15 || std::abs(angle_deg - 270) <= 15) {
			// Calculate the line's y-coordinate at the bottom of the image
			double a = std::cos(theta);
			double b = std::sin(theta);
			double x0 = a * rho;
			double y0 = b * rho;


				// Check if the line's y-coordinate is near the middle row
				if (std::abs(y0 - middle_row) < 0.15 * image_seg_gray.rows) {
					float length = std::abs(rho);
					if (length > longest_line_length) {
						ret = true;
						longest_line_length = length;
						longest_line = line;
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


	// Set the condition of usage
	return ret;
}

void court_segmentation_refinement(cv::Mat& segmentation, cv::Vec2f& line) { 
	//segmentation is a 3 channel image, where green indicates court and black background
	//above the line we can only have background, below the line we can have both background and court

	// Take the info from the line
	float angle_deg = line[1] * 180.0f / CV_PI;
	float slope = -1 / std::tan(line[1]);

	//Count the number of pixels green and black pixels above and below the line
	int green_above = 0, green_below=0, black_above = 0, black_below = 0;

	for (int y = 0; y < segmentation.rows; ++y) {
		for (int x = 0; x < segmentation.cols; ++x) {
			// Check if the pixel is above the line
			if (y < slope * x + line[0]) {
				// Above the line, check if the pixel is green or black
				if (segmentation.at<cv::Vec3b>(y, x) == cv::Vec3b(0, 255, 0)) {
					// Green pixel above the line
					green_above++;
				}
				else {
					// Black pixel above the line
					black_above++;
				}
			}
			else {
				// Below the line, check if the pixel is green or black
				if (segmentation.at<cv::Vec3b>(y, x) == cv::Vec3b(0, 255, 0)) {
					// Green pixel below the line
					green_below++;
				}
				else {
					// Black pixel below the line
					black_below++;
				}
			}
		}
	}

	if (black_above > green_above) {
		//Correct above
		for (int y = 0; y < segmentation.rows; ++y) {
			for (int x = 0; x < segmentation.cols; ++x) {
				// Check if the pixel is above the line
				if (y < slope * x + line[0]) {
					// Above the line, set the pixel to background (black)
					segmentation.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 0, 0);
				}
			}
		}
	}

	else if (black_below > green_below) {
		//Heavy correction
		for (int y = 0; y < segmentation.rows; ++y) {
			for (int x = 0; x < segmentation.cols; ++x) {
				// Check if the pixel is above the line
				if (y < slope * x + line[0]) {
					// Above the line, set the pixel to background (black)
					segmentation.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 0, 0);
				}
				else
				{
					// Below the line, set the pixel to court (green)
					segmentation.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 255, 0);
				}
			}
		}
	}

}
