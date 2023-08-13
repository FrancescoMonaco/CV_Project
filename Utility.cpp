#include "Utility.h"
//Francesco Pio Monaco

//Constants for the utility functions
    //Thresholds for the orange color
const double thres_high_1 = 0.328, thres_low_1 = 0.20, thres_high_2 = 0.16, thres_low_2 = 0.14;

//Implementations
bool hasOrangeColor(const cv::Mat& image, int orange_lower_bound = 120, int orange_upper_bound = 255) {

    // Count variables
    int total_pixels = image.rows * image.cols;
    int orange_pixels = 0;

    for (int y = 0; y < image.rows; y++) {
        for (int x = 0; x < image.cols; x++) {
            cv::Vec3b pixel = image.at<cv::Vec3b>(y, x);
            int blue = pixel[0];
            int green = pixel[1];
            int red = pixel[2];

            if (red >= orange_lower_bound && green >= orange_lower_bound && blue <= orange_upper_bound) {
                orange_pixels++;
            }
        }
    }

    if ( ((static_cast<double>(orange_pixels) / total_pixels) >= thres_low_1 && (static_cast<double>(orange_pixels) / total_pixels) < thres_high_1) ||
         ((static_cast<double>(orange_pixels) / total_pixels) >= thres_low_2 && (static_cast<double>(orange_pixels) / total_pixels) < thres_high_2)
        )
        return true;
    else
        return false;
}