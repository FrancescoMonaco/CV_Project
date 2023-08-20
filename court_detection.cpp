#include"court_detection.h"
#include <fstream>
#include "segmentation.h"

void box_elimination(cv::Mat image, cv::Mat img_out, std::string str)
{	
	//file with boxes 
	std::ifstream file(str);

	//clone the starting image 
	//img_out = image.clone();
	//cv::imshow("image ", img_out);
	//cv::waitKey(0);


	if (file.is_open()) {
		
		std::string line;
		
		//take the string with the position of bounding box
		while (std::getline(file, line)) {
			int x,y,x_last,y_last;

			std::istringstream iss(line);
			
			iss >> x >> y >> x_last >> y_last;

			//delete the box 
			for (int i = x; i <= x_last; i++) {
				for (int j = y; j <= y_last; j++) {
					
					img_out.at<cv::Vec3b>(j, i) = cv::Vec3b(0, 0, 0);
				}
			}
			//cv::imshow("image w/out boxes", img_out);
			//cv::waitKey(0);
		}

	}else {
		std::cout << "error path\n";
	}

	//visualize the image
	cv::imshow("image w/out boxes", img_out);
	cv::waitKey(0);
	
}

void fill_image(cv::Mat image){
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

void court_localization(cv::Mat image){
	
	
	segmentation(image,image);
	
	//start by using canny to localize the lines 
	cv::Mat blur_img;

	cv::GaussianBlur(image,blur_img,cv::Size(9,9),0.5,0.5);

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
	int canny_c = 9;
	std::cout << "Median: " << median << std::endl;
	cv::Mat edges;


	cv::Canny(image, edges, canny_c * median / 2, canny_c * median);
	
	cv::imshow("edges", edges);
	cv::waitKey(0);
}


void color_quantization(cv::Mat image) {

	int numClusters = 2; // Number of desired colors after quantization
	cv::Mat labels, centers;
	cv::TermCriteria criteria=	cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 100, 0.1);
	

	cv::Mat floatImage, clustered ;
	image.convertTo(floatImage, CV_32FC3, 1.0 / 255.0);

	cv::kmeans(floatImage.reshape(1, floatImage.rows * floatImage.cols), numClusters, labels, criteria, 10, cv::KMEANS_PP_CENTERS,centers);

	//replace colors 
	
	

	// Define replacement colors
	cv::Vec3b colors[2];

	colors[0] = cv::Vec3b(255, 0, 0); // Red
	colors[1] = cv::Vec3b(0, 0, 255); // Blue
	clustered = cv::Mat(image.rows, image.cols, CV_8UC3);
	
	for (int i = 0; i < image.cols * image.rows; i++) {
		
		int el=labels.at<int>(0,i);
		//std::cout << el<<std::endl;
		if (el == 1) {
			//std::cout << "eccolo\n";
		}
		clustered.at<cv::Vec3b>(i / image.cols, i % image.cols)= colors[el] ;
	
	}

	//cv::Mat converted;
	//clustered.convertTo(converted, CV_8U);
	//clustered.convertTo(clustered, CV_8U);
	//cv::imshow("Original Image", image);
	cv::imshow("Quantized Image", clustered);
	cv::waitKey(0);

}