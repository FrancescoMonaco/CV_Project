#include "../headers/Utility.h"
// Utility.cpp : Francesco Pio Monaco

//Constants for the utility functions
    //Thresholds for the orange color
const double thres_high_1 = 0.328, thres_low_1 = 0.20, thres_high_2 = 0.16, thres_low_2 = 0.14;
	//Multiplicators for canny and the heat diffusion
const int canny_c = 9, alpha = 1; const double lambda = 1;


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

void removeUniformRect(std::vector<cv::Rect>& rects, cv::Mat image, int threshold)
{
    for (size_t i = 0; i < rects.size(); i++)
    {
        resizeRect(rects[i], image);
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
    //if two rectangles overlap on the y axis, merge them into a single rectangle that contains both
    for (size_t i = 0; i < rects.size(); i++)
    {
        for (size_t j = i + 1; j < rects.size(); j++)
        {
			cv::Rect r1 = rects[i];
			cv::Rect r2 = rects[j];
            if (r1.y < r2.y)
            {
                if (r1.y + r1.height > r2.y)
                {
                    if (r1.x < r2.x)
                    {
                        if (r1.x + r1.width > r2.x)
                        {
							//r1 contains r2
							rects.erase(rects.begin() + j);
							j--;
						}
					}
                    else
                    {
                        if (r2.x + r2.width > r1.x)
                        {
							//r2 contains r1
							rects.erase(rects.begin() + i);
							i--;
							break;
						}
					}
				}
			}
            else
            {
                if (r2.y + r2.height > r1.y)
                {
                    if (r1.x < r2.x)
                    {
                        if (r1.x + r1.width > r2.x)
                        {
							//r1 contains r2
							rects.erase(rects.begin() + j);
							j--;
						}
					}
                    else
                    {
                        if (r2.x + r2.width > r1.x)
                        {
							//r2 contains r1
							rects.erase(rects.begin() + i);
							i--;
							break;
						}
					}
				}
			}
		}
	}
}

void cleanRectangles(std::vector<cv::Rect>& rects, cv::Mat image)
{
    cv::Mat mskd = computeDiffusion(image);
    removeUniformRect(rects, mskd, 30);
    removeFlatRect(rects);
    mergeOverlapRect(rects, 0);
}

cv::Mat computeDiffusion(cv::Mat image)
{
    cv::Mat test, edges;
    //Do a strong blur before canny
    cv::GaussianBlur(image, test, cv::Size(7, 7), 0.4, 0.4);
    //Compute the gradient magnitude of the image
    cv::Mat grad_x, grad_y;
    cv::Mat abs_grad_x, abs_grad_y, test_grad;
    cv::Sobel(test, grad_x, CV_16S, 1, 0, 3, 1, 0, cv::BORDER_DEFAULT);
    cv::Sobel(test, grad_y, CV_16S, 0, 1, 3, 1, 0, cv::BORDER_DEFAULT);
    cv::convertScaleAbs(grad_x, abs_grad_x);
    cv::convertScaleAbs(grad_y, abs_grad_y);
    cv::addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, test_grad);
    //Compute the median of the gradient magnitude
    cv::Scalar mean, stddev;
    cv::meanStdDev(test_grad, mean, stddev);
    double median = mean[0];

    cv::Canny(test, edges, canny_c * median / 2, canny_c * median);

    int iterations = 40;
    cv::Mat diffusedImage = edges.clone();

    //Apply diffusion iteratively
    for (int iter = 0; iter < iterations; ++iter) {
        cv::Mat newDiffusedImage = diffusedImage.clone();

        for (int y = 1; y < diffusedImage.rows - 1; ++y) {
            for (int x = 1; x < diffusedImage.cols - 1; ++x) {
                // Apply heat diffusion equation
                double newValue = diffusedImage.at<uchar>(y, x) + alpha * (
                    diffusedImage.at<uchar>(y - 1, x) + diffusedImage.at<uchar>(y + 1, x) +
                    diffusedImage.at<uchar>(y, x - 1) + diffusedImage.at<uchar>(y, x + 1) -
                    4 * diffusedImage.at<uchar>(y, x)
                    );

                newDiffusedImage.at<uchar>(y, x) = cv::saturate_cast<uchar>(newValue);
            }
        }

        diffusedImage = newDiffusedImage;
    }

    //Apply a max kernel on diffusedImage
    cv::Mat maxKernel = cv::Mat::ones(3, 3, CV_8U);
    cv::Mat maxImage;
    cv::dilate(diffusedImage, maxImage, maxKernel);

    cv::Mat mask;
    cv::threshold(maxImage, mask, 1, 255, cv::THRESH_BINARY);

    // Apply the mask to the original image
    cv::Mat maskedImage;
    test.copyTo(maskedImage, mask);
    return maskedImage;
}

std::vector<std::vector<cv::Rect>> reshapeBB(std::vector<BoundingBox> bbs, int NUM_IMAGES)
{
    std::vector<std::vector<cv::Rect>> processedData2;
    for (int i = 0; i < NUM_IMAGES; i++)
    {
		std::vector<cv::Rect> processedData;
        for (int j = 0; j < bbs.size(); j++)
        {
            if (bbs[j].fileNum == i)
            {
				cv::Rect r = cv::Rect(bbs[j].x1, bbs[j].y1, bbs[j].width, bbs[j].height);
				processedData.push_back(r);
			}
		}
		processedData2.push_back(processedData);
	}

    return processedData2;
}

void removeFlatRect(std::vector<cv::Rect>& rects)
{
    //for each rectangle, check if it is flat and remove it if it is
    for (size_t i = 0; i < rects.size(); i++)
    {
		cv::Rect r = rects[i];
        if (r.width > 2.5 * r.height)
        {
			rects.erase(rects.begin() + i);
			i--;
		}
	}
}

void resizeRect(cv::Rect& rect, cv::Mat image) {
    //Resize the rectangle to fit the image
    if (rect.x < 0)
    {
		rect.width += rect.x;
		rect.x = 0;
	}
    if (rect.y < 0)
    {
		rect.height += rect.y;
		rect.y = 0;
	}
    if (rect.x + rect.width > image.cols)
    {
		rect.width = image.cols - rect.x;
	}
    if (rect.y + rect.height > image.rows)
    {
		rect.height = image.rows - rect.y;
	}
}
