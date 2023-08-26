#include"court_detection.h"
#include "peer_filter.h"
#include "color_quantization.h"
#include <algorithm>

void box_elimination(cv::Mat image, cv::Mat img_out, std::string str)
{	
	//file with boxes 
	std::ifstream file(str);

	if (file.is_open()) {
		
		std::string line;
		
		//take the string with the position of bounding box
		while (std::getline(file, line)) {
			int x,y,w,h;

			std::istringstream iss(line);
			
			iss >> x >> y >> w >> h;
			
			/*
			std::cout << x << std::endl;
			std::cout << y << std::endl;
			std::cout << w << std::endl;
			std::cout << h << std::endl;
			*/

			//delete the box 
			for (int i = x; i <= x+w; i++) {
				for (int j = y; j <= y+h; j++) {
					
					img_out.at<cv::Vec3b>(j, i) = cv::Vec3b(0, 0, 0);

				}
			}
		}

	}else {

		std::cout << "error path\n";

	}

	//visualize the image
	//cv::imshow("image w/out boxes", img_out);
	//cv::waitKey(0);
	
}

void fill_image(cv::Mat& image){
	//start filling the image
	
	
	for (int i = 0; i < image.rows; i++) {
		for (int j = 0; j < image.cols; j++) {
			//i fill the holes left by the remove of boxes
			if (image.at<cv::Vec3b>(i, j)==cv::Vec3b(0,0,0)) {
				cv::Vec3b color = image.at<cv::Vec3b>(i , j-1);
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


void color_quantization(cv::Mat image,cv::Mat& img_out) {



	
	int numClusters = 3; // Number of desired colors after quantization
	cv::Mat labels, centers;
	cv::TermCriteria criteria=	cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 100, 0.1);
	
	std::vector<cv::Vec3b> vec;
	cv::Mat mask(image.rows,image.cols, CV_8UC1);
	std::vector<cv::Point> pixel_positions;

	for (int i = 0; i < image.rows; i++) {
		for (int j = 0; j < image.cols; j++) {
			
			if (image.at<cv::Vec3b>(i, j) == cv::Vec3b(0,0,0)) {
				
				mask.at<uchar>(i, j) = 0;
			
			}else{

				vec.push_back(image.at<cv::Vec3b>(i, j));
				mask.at<uchar>(i, j) = 1;
				pixel_positions.push_back(cv::Point(j, i)); // Store pixel positions

			}
		}
	}



	// Convert Vec3b data to a format suitable for K-means
	cv::Mat flattened_data(vec.size() , 3, CV_32F);
	
	for (size_t i = 0; i < vec.size(); ++i) {
		flattened_data.at<float>(i, 0) = vec[i][0];
		flattened_data.at<float>(i, 1) = vec[i][1];
		flattened_data.at<float>(i, 2) = vec[i][2];
		flattened_data.at<float>(i,4) = pixel_positions[i].x;         // X
		flattened_data.at<float>(i,5) = pixel_positions[i].y;         // Y
	}

	cv::normalize(flattened_data, flattened_data, 0, 1, cv::NORM_MINMAX);

	cv::Mat floatImage, clustered ;
	image.convertTo(floatImage, CV_32FC3, 1.0 / 255.0);
	cv::Mat flat = floatImage.reshape(1, floatImage.rows * floatImage.cols);

	cv::kmeans(flattened_data, numClusters, labels, criteria, 10, cv::KMEANS_PP_CENTERS,centers);


	
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

			if (mask.at<uchar>(i,j)==1) {

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
	//cv::imshow("Quantized Image", clustered);
	//cv::waitKey(0);
	img_out = clustered.clone();
	
	lines_detector(img_out);


}

 
void field_distinction(cv::Mat image_box,cv::Mat clustered,cv::Mat& segmented_field) {
	
	//colour used to distict field and background
	cv::Vec3b green(0,255,0);
	
	cv::Mat mask1,mask2, mask3;
	cv::inRange(clustered, cv::Vec3b(255, 0, 0), cv::Vec3b(255, 0, 0),mask1);
	
	cv::inRange(clustered, cv::Vec3b(0, 0,255), cv::Vec3b(0, 0, 255), mask2);
	
	cv::inRange(clustered, cv::Vec3b(0, 255,0), cv::Vec3b(0, 255, 0), mask3);


	int pixel_count1 = cv::countNonZero(mask1);
	int pixel_count2 = cv::countNonZero(mask2);
	int pixel_count3 = cv::countNonZero(mask3);

	int larger=std::max(pixel_count1,pixel_count2);
	larger = std::max(larger, pixel_count3);

	//for each region i calculate the standar deviation
	standard_deviation(image_box,mask1);
	standard_deviation(image_box,mask2);
	standard_deviation(image_box,mask3);


	if (pixel_count1 ==larger) {
		std::cout << "1\n";
		segmented_field.setTo(green, mask1);
		segmented_field.setTo(cv::Vec3b(0,0,0), mask2);
		segmented_field.setTo(cv::Vec3b(0,0,0), mask3);
	}
	else if(pixel_count2==larger){
		std::cout << "2\n";
		segmented_field.setTo(green, mask2);
		segmented_field.setTo(cv::Vec3b(0, 0, 0), mask1);
		segmented_field.setTo(cv::Vec3b(0, 0, 0), mask3);

	}
	else {
		std::cout << "3\n";
		segmented_field.setTo(green, mask3);
		segmented_field.setTo(cv::Vec3b(0, 0, 0), mask1);
		segmented_field.setTo(cv::Vec3b(0, 0, 0), mask2);

	}
	cv::imshow("segmented field", segmented_field);
	cv::waitKey(0);
}

std::vector<double> standard_deviation(cv::Mat box_image, cv::Mat mask) {

	double sumr, sumg, sumb;
	int count;
	
	// calculate the mean of each channel 
	for (int i = 0; i < mask.rows; i++) {
		for (int j = 0; j < mask.cols;j++) {

			if (mask.at<uchar>(i, j) ==1 ) {  
				sumr += box_image.at<cv::Vec3b>(i, j)[0];
				sumg += box_image.at<cv::Vec3b>(i, j)[1];
				sumb += box_image.at<cv::Vec3b>(i, j)[2];
				count++;
			}
		}
	}

	sumr= sumr / count;
	sumg= sumg / count;
	sumb= sumb / count;


	//calculate the standard deviation
	
	double sum_squared_diffr = 0.0;
	double sum_squared_diffg = 0.0;
	double sum_squared_diffb = 0.0;
	
	for (int y = 0; y < mask.rows; ++y) {
		for (int x = 0; x < mask.cols; ++x) {
			
			if (mask.at<uchar>(y, x) ==1 ) {  // Assuming binary mask
				double diffr = box_image.at<cv::Vec3b>(y, x)[0] - sumr;
				double diffg = box_image.at<cv::Vec3b>(y, x)[1] - sumg;
				double diffb = box_image.at<cv::Vec3b>(y, x)[2] - sumb;
				
				sum_squared_diffr += diffr * diffr;
				sum_squared_diffg += diffg * diffg;
				sum_squared_diffb += diffb * diffb;
				
				
			}
		}
	}

	double sdr=std::sqrt(sum_squared_diffr / count);
	double sdg=std::sqrt(sum_squared_diffg / count);
	double sdb=std::sqrt(sum_squared_diffb / count);
	
	//create the vector containing the standard deviations for each channel

	std::vector<double> standars;
	standars.push_back(sdr);
	standars.push_back(sdg);
	standars.push_back(sdb);

	return standars;

}





//localize the field lines by using hough transform not used 
void lines_detector(cv::Mat image) {


	cv::Mat edge;

	court_localization(image,edge);

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
