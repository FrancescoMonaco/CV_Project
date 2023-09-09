#include "../headers/player_detection.h"
#include <cmath>

void player_segmentation(cv::Mat image, cv::Mat& seg_image, std::string str) {


	std::ifstream file(str);
	cv::Mat mask = image.clone();
	create_mask(image, mask, str);

	if (file.is_open()) {

		std::string line;

		//take the string with the position of bounding box
		while (std::getline(file, line)) {
			std::vector<int> parameters;

			int x, y, w, h;

			std::istringstream iss(line);

			iss >> x >> y >> w >> h;

			cv::Mat img_out(h, w, CV_8UC3);

			parameters.push_back(x);
			parameters.push_back(y);
			parameters.push_back(w);
			parameters.push_back(h);
			//isolate the box CHECK

			cv::Mat mask_temp = mask.clone();
			for (int j = y; j < y + h; j++) {
				for (int i = x; i < x + w; i++) {

					img_out.at<cv::Vec3b>(j - y, i - x) = image.at<cv::Vec3b>(j, i);
					mask_temp.at<cv::Vec3b>(j, i) = image.at<cv::Vec3b>(j, i);

				}
			}

			/*cv::imshow("blurred image", img_out);
			cv::waitKey(0);*/

			cv::Mat blur;
			cv::GaussianBlur(img_out, blur, cv::Size(5, 5), 0.8, 0.8);

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


			cv::Mat edges;


			cv::Canny(img_grey, edges, canny_c * median / 4, canny_c * median / 2);

			/*cv::imshow("canny ", edges);
			cv::waitKey(0);*/


			// Find contours in the binary image
			std::vector<std::vector<cv::Point>> contours;
			cv::findContours(edges, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

			// Create a copy of the original image to draw the detected lines
			cv::Mat output_image(img_grey.size(), CV_8UC1, cv::Scalar(0, 0, 0));//temp

			// Filter and draw only the long contours (lines)
			int min_contour_length = 25; // Set your desired minimum contour (line) length

			for (int i = 0; i < contours.size(); i++) {

				if (cv::arcLength(contours[i], true) >= min_contour_length) {
					cv::drawContours(output_image, contours, static_cast<int>(i), cv::Scalar(255, 255, 255), 1); // Set thickness to 1
				}

			}

			// Display the resulting image with the detected lines
			/*cv::imshow("Detected Lines", output_image);
			cv::waitKey(0);*/


			//close the lines found out by using the clustering and after removing the less important 
			close_lines(edges);

			/*cv::imshow(" ", edges);
			cv::waitKey(0);*/

			create_lines(edges, edges);

			/*cv::imshow(" ",edges);
			cv::waitKey(0);*/

			//i use this function to color inside the figures
			fill_segments(edges);

			/*cv::imshow(" ", edges);
			cv::waitKey(0);*/

			cv::Mat cluster;

			//clustering(mask_temp, cluster);

			//super_impose(cluster, edges, parameters);

			//create the final mask for segmentation

			//for (int j = y; j < y + h; j++) {
			//	for (int i = x; i < x + w; i++) {

			//		//make the union of bodies in different boxes
			//		if (seg_image.at<uchar>(j, i) != 255) {
			//			seg_image.at<uchar>(j, i) = edges.at<uchar>(j - y, i - x);
			//		}
			//		else {
			//			continue;
			//		}
			//	}
			//}
			//
			//	cv::imshow("mask", mask_temp);
			//	cv::waitKey(0);
			

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

	cv::Mat element = getStructuringElement(cv::MORPH_ELLIPSE
		, cv::Size(morph_size, morph_size));

	cv::Mat img_out;
	morphologyEx(edge_image, img_out, cv::MORPH_GRADIENT, element, cv::Point(-1, -1), 2);


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
		drawContours(edge_image, contours, idx, color, cv::FILLED, 4, hierarchy);
	}

	//cv::imshow("Components", edge_image);
	//cv::waitKey(0);
}




void clustering(cv::Mat image, cv::Mat& cluster) {


	int numClusters = 15; // Number of desired colors after quantization
	cv::Mat labels, centers;
	cv::TermCriteria criteria = cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 10, 0.1);

	cv::Mat image_box;
	cv::cvtColor(image, image_box, cv::COLOR_BGR2HSV);

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
	cv::Mat flattened_data(vec.size(), 50, CV_32F);

	for (size_t i = 0; i < vec.size(); ++i) {
		flattened_data.at<float>(i, 0) = vec[i][0];
		flattened_data.at<float>(i, 1) = vec[i][1];
		flattened_data.at<float>(i, 2) = vec[i][2];


	}

	cv::normalize(flattened_data, flattened_data, 0, 1, cv::NORM_MINMAX);

	//cv::Mat flat = image_box.reshape(1, image_box.cols * image_box.rows);
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


	cv::imshow("clustering", cluster);
	cv::waitKey(0);





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

	//cv::imshow("mask", mask);
	//cv::waitKey(0);
}



void create_lines(cv::Mat edges, cv::Mat& output_edges) {

	std::vector<cv::Point> starters;
	std::vector<cv::Point> terminators;


	cv::imshow("first edges", edges);
	//cv::waitKey(0);*/

	bool start = false;

	starters.clear();
	terminators.clear();

	for (int j = 0; j < edges.cols; j++) {

		uchar pixel = edges.at<uchar>(0, j);

		if (!start && pixel == 255 && j + 1 != edges.cols && edges.at<uchar>(0, j + 1) != 255) {

			starters.push_back(cv::Point(j, 0));
			start = true;
		}

		else if (start && pixel == 255) {

			start = false;
			terminators.push_back(cv::Point(j, 0));
			//std::cout << "found\n";
		}

	}

	if (start) {
		starters.pop_back();

	}


	//create the line
	for (int i = 0; i < starters.size(); i++) {

		//	std::cout<< starters[i] <<std::endl;
			//take stating and ending point
		cv::Point starte = starters[i];
		cv::Point end = terminators[i];

		if (end.x - starte.x > edges.rows/5) {

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



	//---------------

	start = false;

	starters.clear();
	terminators.clear();

	int n_rows = edges.rows;

	for (int j = 0; j < edges.cols; j++) {

		uchar pixel = edges.at<uchar>(n_rows - 1, j);

		if (!start && pixel == 255 && j + 1 != edges.cols && edges.at<uchar>(n_rows - 1, j + 1) != 255) {

			starters.push_back(cv::Point(j, n_rows - 1));
			start = true;

		}

		else if (start && pixel == 255) {
			start = false;
			terminators.push_back(cv::Point(j, n_rows - 1));
			//std::cout << "found\n";
		}

	}

	if (start) {
		starters.pop_back();

	}



	//create the line
	for (int i = 0; i < starters.size(); i++) {

		//take stating and ending point
		cv::Point starte = starters[i];
		cv::Point end = terminators[i];

		if (end.x - starte.x < edges.rows/10 && end.x - starte.x> edges.rows/5) {
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






	//---------------------

	start = false;

	starters.clear();
	terminators.clear();



	for (int j = 0; j < edges.rows; j++) {

		uchar pixel = edges.at<uchar>(j, 0);

		if (!start && pixel == 255 && j + 1 != edges.rows && edges.at<uchar>(j + 1, 0) != 255) {

			starters.push_back(cv::Point(0, j));
			start = true;

		}

		else if (start && pixel == 255) {

			start = false;
			terminators.push_back(cv::Point(0, j));
			//std::cout << "found\n";
		}

	}

	if (start) {
		starters.pop_back();

	}



	//std::cout << starters;

	//create the line
	for (int i = 0; i < starters.size(); i++) {


		cv::Point starte = starters[i];
		cv::Point end = terminators[i];

		//std::cout << end.y - starte.y<<std::endl;
		if ((end.y - starte.y) < edges.rows/15 || (end.y - starte.y) > edges.rows/10) {

		}
		else {
			for (int j = starte.y; j < end.y; j++) {

				//std::cout << end.y - starte.y << std::endl;
				edges.at<uchar>(j, 0) = 255;
				edges.at<uchar>(j, 1) = 255;
				edges.at<uchar>(j, 2) = 255;
				edges.at<uchar>(j, 3) = 255;
				edges.at<uchar>(j, 4) = 255;

			}
		}

	}

	
	//-------------------------------
	
	
		/**/
		start = false;
	
		starters.clear();
		terminators.clear();
	
		int col = edges.cols;
	
		for (int j = 0; j < edges.rows; j++) {
	
			uchar pixel = edges.at<uchar>(j, col-1);
	
			if (!start && pixel == 255 && j + 1 != edges.rows && edges.at<uchar>(j + 1, col-1) != 255) {
	
				starters.push_back(cv::Point(col-1, j));
				start = true;
	
			}
	
			else if (start && pixel == 255) {
	
				start = false;
				terminators.push_back(cv::Point(col-1, j));
				//std::cout << "found\n";
			}
	
		}
	
		if (start) {
			starters.pop_back();
	
		}
	
	
	
		//std::cout << starters;
	
		//create the line
		for (int i = 0; i < starters.size(); i++) {
	
	
			cv::Point starte = starters[i];
			cv::Point end = terminators[i];
	
			if (end.y - starte.y > edges.cols/5) {
	
			}
			else {
				std::cout << "In th else";
				for (int j = starte.y; j < end.y; j++) {
	
					edges.at<uchar>(j, col-1) = 255;
					edges.at<uchar>(j, col-2) = 255;
					edges.at<uchar>(j, col-3) = 255;
					edges.at<uchar>(j, col-4) = 255;
					edges.at<uchar>(j, col-5) = 255;
	
				}
			}
	
		}

	output_edges = edges.clone();
	cv::imshow("new edges", edges);
	cv::waitKey(0);



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


	double n_nonzeros = cv::countNonZero(mask);
	double tot = mask.cols * mask.rows;
	double n_zeros = tot - n_nonzeros;

	double more = 0.0;

	if (n_nonzeros / tot > 0.75) {

		if (x + w + 20 < clustering.cols) {

			w += 20;

			cv::Mat paddedImage(mask.rows, mask.cols + 20, mask.type(), cv::Vec3b(0, 0, 0));
			mask.copyTo(paddedImage(cv::Rect(0, 0, mask.cols, mask.rows)));
			mask = paddedImage.clone();
			//more = -0.1;
			n_zeros = n_zeros + (20 * mask.rows);
		}

		if (x - 20 > 0) {
			x = x - 20;
			cv::Mat paddedImage(mask.rows, mask.cols + 20, mask.type(), cv::Vec3b(0, 0, 0));
			mask.copyTo(paddedImage(cv::Rect(20, 0, mask.cols, mask.rows)));
			mask = paddedImage.clone();
			//more = -0.1;
			n_zeros = n_zeros + (20 * mask.rows);
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


	cv::imshow(" ", box_superimpose);
	cv::waitKey(0);

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
					//cv::imshow("io osno", temp);
					//cv::waitKey(0);
					combinedVector.push_back(std::pair(pixel, color));
					colors.push_back(color);
				}

			}

		}
	}


	double num_labels = colors.size();
	std::sort(combinedVector.begin(), combinedVector.end(), sortbysec);
	//i take only the pixel who are the most out 

	for (int z = 0; z < colors.size(); z++) {
		double n_elements = combinedVector[z].first;

		double fr = n_elements / n_zeros;
		//std::cout << fr << std::endl;
		double tr = 1 / num_labels;
		//std::cout << tr << std::endl;
		//skip
		if (fr < tr + more) {
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
			cv::imshow(" ", box_superimpose);
			cv::waitKey(0);
		}

	}

	/*cv::imshow("final", box_superimpose);
	cv::waitKey(0);
	*/
	cv::Mat final_seg, inversion;
	cv::inRange(box_superimpose, cv::Vec3b(0, 0, 0), cv::Vec3b(0, 0, 0), inversion);

	cv::bitwise_not(inversion, final_seg);
	cv::imshow("", final_seg);
	cv::waitKey(0);

	mask = final_seg.clone();
	remove_components(mask);
}
void remove_components(cv::Mat& mask) {

	// Create a labeled image to store connected components
	cv::Mat labeledImage;
	int numLabels = cv::connectedComponents(mask, labeledImage, 8, CV_32S);

	// Create a vector to store the pixel counts for each region
	std::vector<int> regionPixelCounts(numLabels, 0);

	// Iterate through the labeled image and count pixels for each label
	for (int y = 0; y < labeledImage.rows; y++) {
		for (int x = 0; x < labeledImage.cols; x++) {
			int label = labeledImage.at<int>(y, x);

			if (label > 0) {
				regionPixelCounts[label]++;
			}

		}
	}

	//
	//std::vector<cv::Vec3b> colors(numLabels);
	//for (int label = 0; label < numLabels; ++label) {
	//	colors[label] = cv::Vec3b(rand() % 256, rand() % 256, rand() % 256);
	//}


	//// Create a color image from the labeled image
	//cv::Mat coloredLabels(labeledImage.size(), CV_8UC3);
	//for (int i = 0; i < labeledImage.rows; ++i) {
	//	for (int j = 0; j < labeledImage.cols; ++j) {
	//		int label = labeledImage.at<int>(i, j);
	//		coloredLabels.at<cv::Vec3b>(i, j) = colors[label];
	//	}
	//}
	double tot = mask.cols * mask.rows;

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

	cv::imshow("final mask", mask);
	cv::waitKey(0);

}
