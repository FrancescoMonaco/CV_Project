#include "player_detection.h"
#include<fstream>
#include "court_detection.h"

void player_segmentation(cv::Mat image, cv::Mat seg_image, std::string str){
	
	
	std::ifstream file(str);

	//clone the starting image 
	//img_out = image.clone();
	//cv::imshow("image ", img_out);
	//cv::waitKey(0);

	
	if (file.is_open()) {

		std::string line;

		//take the string with the position of bounding box
		while (std::getline(file, line)) {
			int x, y, x_last, y_last;

			std::istringstream iss(line);
			
			iss >> x >> y >> x_last >> y_last;

			cv::Mat img_out(y_last-y, x_last - x, CV_8UC3);
			//isolate the box CHECK
			
			for (int j = y; j < y_last; j++) {
				for (int i = x; i < x_last; i++) {

					img_out.at<cv::Vec3b>(j-y, i-x) = image.at<cv::Vec3b>(j,i);
				}
			}
			//cv::imshow("box", img_out);
			//cv::waitKey(0);

			color_quantization(img_out);

			//start segmentation with canny

	 
			cv::Mat img_grey;
			cvtColor(img_out, img_grey, cv::COLOR_BGR2GRAY);
			

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
			int canny_c = 9;
			//std::cout << "Median: " << median << std::endl;
			cv::Mat edges;


			cv::Canny(img_grey, edges, canny_c * median / 4,canny_c * median/2);

			cv::imshow("edges", edges);
			cv::waitKey(0);


			//apply heat diffusion
			heat_diffusion(edges);

		}

	}
	else {
		std::cout << "error path\n";
	}

	
	
	
}


void heat_diffusion(cv::Mat canny_image){
	// Read the Canny edge image
	cv::Mat E = canny_image.clone();

	// Normalize the Canny image to the range [0, 1]
	E.convertTo(E, CV_32F, 1.0 / 255.0);

	// Define image size
	int width = E.cols;
	int height = E.rows;

	// Create an image with initial condition
	cv::Mat I(height, width, CV_32F);
	I = 1.0 - E;

	// Define constants
	float dt = 0.1; // Time step
	float alpha = 0.1; // Diffusion coefficient

	// Create a window for visualization
	cv::namedWindow("Heat Diffusion", cv::WINDOW_NORMAL);

	// Time loop
	for (float t = 0; t <100; t += dt) {
		// Compute Laplacian
		cv::Mat laplacian;
		cv::Laplacian(I, laplacian, CV_32F);

		// Apply heat diffusion equation
		I = I + alpha * dt * laplacian;

		// Apply boundary condition
		I.setTo(0, I == 1);

		// Display the image
		
	}

	cv::imshow("Heat Diffusion", I);
	cv::waitKey(0); // Delay for visualization
	cv::destroyAllWindows();


}