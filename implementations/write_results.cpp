#include "../headers/write_results.h"

void write_segmentation_results(cv::Mat segmented_image,cv::Mat saved_bin, std::string save){
	
	cv::Vec3b black(0,0,0);
	cv::Vec3b green(0,255,0);
	cv::Vec3b red(255,0,0);
	cv::Vec3b blue(0,0,255);

	for (int i = 0; i < segmented_image.rows; i++) {
		for (int j = 0; j < segmented_image.cols; j++) {

			cv::Vec3b color = segmented_image.at<cv::Vec3b>(i,j);
			if(color==black){
				saved_bin.at<uchar>(i, j)=0 ;
			
			}else if (color == green) {
				saved_bin.at<uchar>(i, j) =3 ;
			}else if (color == red) {
				saved_bin.at<uchar>(i, j) = 2;
			}else if (color == blue) {
				saved_bin.at<uchar>(i, j) = 1;
			}
			else {
				std::cout << "some error occurred during the segmentation\n";
				break;

				//saved_bin.at<uchar>(i, j) = 0;
			}
		}
	}
	cv::imwrite(save,saved_bin);


}
