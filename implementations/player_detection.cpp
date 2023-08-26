#include "../headers/player_detection.h"

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
			int x, y, w, h;

			std::istringstream iss(line);
			
			iss >> x >> y >> w >> h;

			cv::Mat img_out( h, w, CV_8UC3);
			//isolate the box CHECK
			
			for (int j = y; j < y+h; j++) {
				for (int i = x; i < x+w; i++) {

					img_out.at<cv::Vec3b>(j-y, i-x) = image.at<cv::Vec3b>(j,i);
				}
			}
			cv::imshow("box", img_out);
			cv::waitKey(0);
			/*cv::Mat clustered;
;			color_quantization(img_out,clustered);
*/
			//start segmentation with canny

			cv::Mat img_grey;
			cvtColor(img_out, img_grey, cv::COLOR_BGR2GRAY);
			//smooth the image in a very strong way
			cv::GaussianBlur(img_grey, img_grey, cv::Size(7, 7), 0, 0, cv::BORDER_DEFAULT);
			// Calculate the minimum and maximum pixel values in the image
			double minVal, maxVal;
			cv::minMaxLoc(img_grey, &minVal, &maxVal);

			// Apply histogram stretching for contrast adjustment
			cv::Mat stretchedImage;
			double alpha = 255.0 / (maxVal - minVal);
			double beta = -minVal * alpha;
			img_grey.convertTo(stretchedImage, -1, alpha, beta);


			cv::Mat grad_x, grad_y;
			cv::Mat abs_grad_x, abs_grad_y, test_grad;

			cv::Sobel(stretchedImage, grad_y, CV_16S, 0, 1, 3, 1, 0, cv::BORDER_DEFAULT);
			cv::Sobel(stretchedImage, grad_x, CV_16S, 1, 0, 3, 1, 0, cv::BORDER_DEFAULT);
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


			cv::Canny(stretchedImage, edges, canny_c * median / 4,canny_c * median/2);

			cv::imshow("edges", edges);
			cv::waitKey(0);

			close_lines(edges);
			

			fill_segments(edges);
			//apply heat diffusion
			//heat_diffusion(edges);
			//cv::Mat seg_image;
			//segmentation(img_out,seg_image );


			cv::destroyAllWindows();
		}

	}
	else {
		std::cout << "error path\n";
	}

	
	
	
}




void close_lines(cv::Mat& edge_image){
	
	int morph_size = 3;

	cv::Mat element = getStructuringElement(cv::MORPH_RECT, cv::Size(morph_size ,  morph_size));
	
	cv::Mat img_out;
	morphologyEx(edge_image, img_out, cv::MORPH_CLOSE, element, cv::Point(-1,-1), 3);
	
	cv::imshow("dilation+erosion", img_out);
	cv::waitKey(0);
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
		cv::Scalar color(255,255,255);
		drawContours(dst, contours, idx, color, cv::FILLED, 8, hierarchy);
	}
	
	cv::imshow("Components", dst);
	cv::waitKey(0);
}

void create_segmented_image(cv::Mat segmeted_filed, cv::Mat segmented_player,std::vector<int> box_coordinates, std::string save){



}

