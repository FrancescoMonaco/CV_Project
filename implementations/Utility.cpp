#include "../headers/Utility.h"
// Utility.cpp : Francesco Pio Monaco

//Constants for the utility functions
    //Thresholds for the orange color
const double thres_high_1 = 0.328, thres_low_1 = 0.20, thres_high_2 = 0.16, thres_low_2 = 0.14;

//Implementations
bool hasOrangeColor(const cv::Mat& image, int orange_lower_bound, int orange_upper_bound) {

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

void removeUniformRect(std::vector<cv::Rect>& rects, cv::Mat image,int threshold)
{
    for (size_t i = 0; i < rects.size(); i++)
    {
        cv::Rect r = rects[i];
        //Calculate color variation within the detected rectangle
        cv::Mat detected_region = image(cv::Rect(r.x, r.y, r.width, r.height));
        cv::Scalar mean_color, stddev_color;
        cv::meanStdDev(detected_region, mean_color, stddev_color);
        double color_variation = stddev_color[0]; // Assuming grayscale image
        if (color_variation > threshold)
        {
            //Put the rectangle in the vector only if it doesn't contain a semi uniform color
            rects[i] = r;
        }
        else
        {
            rects.erase(rects.begin() + i);
            i--;
        }
    }
}

void mergeOverlapRect(std::vector<cv::Rect>& rects, int threshold)
{
    for (size_t i = 0; i < rects.size(); i++)
    {
        cv::Rect r = rects[i];
        for (size_t j = 0; j < rects.size(); j++)
        {
            if (j != i)
            {
                cv::Rect r2 = rects[j];
                if ((r & r2).area() > threshold * r.area() || (r & r2).area() > threshold * r2.area())
                {
                    r |= r2;
                    //If the merged height is greater than 2.5 times the width, then don't put it in the vector but erase the other two
                    if (r.height > 2.5 * r.width)
                    {
                        rects.erase(rects.begin() + i);
                        rects.erase(rects.begin() + j);
                        i--;
                        j--;
                    }
                    else
                    {
                        rects[i] = r;
                        rects.erase(rects.begin() + j);
                        j--;
                    }
                }
            }
        }
    }
}
