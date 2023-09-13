#include "../headers/Utility.h"

// Utility_Class_Rect.cpp : Francesco Pio Monaco

//***Constants for the utility functions
	//Multiplicators for canny and the heat diffusion
const int canny_c = 9, alpha = 1; const double lambda = 1;
     //Colors for the segmentation
std::vector<cv::Vec3b> colors = { cv::Vec3b(0, 0, 255), cv::Vec3b(255, 0, 0) };
	 //Threshold for the blackness of the rectangle
const int BLACK_THRESH = 35;

//***Implementations

void removeUniformRect(std::vector<cv::Rect>& rects, cv::Mat image, int threshold)
{
    //for each rectangle, remove those whose heigth is smaller or equal than two times the width
    for (size_t i = 0; i < rects.size(); i++)
    {
		cv::Rect r = rects[i];
        if (1.7* r.height <=  r.width)
        {
			rects.erase(rects.begin() + i);
			i--;
		}

        //check how black is the rectangle, if it is too black, remove it
        cv::Mat roi = image(r);
        cv::Scalar mean, stddev;
        cv::meanStdDev(roi, mean, stddev);
        double mean_val = mean[0];
        if (mean_val < BLACK_THRESH)
        {
            rects.erase(rects.begin() + i);
            i--;
        }
	}
}

void mergeOverlapRect(std::vector<cv::Rect>& rects, int threshold)
{
    //for each rectangle, check if they overlap on the top or on the bottom, if above a certain threshold, 
    // merge them into a single rectangle that contains both
    for (size_t i = 0; i < rects.size(); i++)
    {
        cv::Rect r = rects[i];
        for (size_t j = i + 1; j < rects.size(); j++)
        {
                //check the area of the intersection, if it's above a certain threshold, merge the rectangles
            cv::Rect r2 = rects[j];
            cv::Rect intersection = r & r2;
            if (intersection.area() > threshold)
            {
				cv::Rect newRect = r | r2;
				rects[i] = newRect;
				rects.erase(rects.begin() + j);
				j--;
			}
        }
    }
    
}

void cleanRectangles(std::vector<cv::Rect>& rects, cv::Mat image)
{
    bool merge = true;
    //If there are many or too few rectangles merging with the same threshold can be a problem
    if (rects.size() >= 11 || (rects.size() > 2 && rects.size() < 6))
		merge = false;

    //Compute the diffusion of the image
    cv::Mat mskd = computeDiffusion(image);

    //Remove rectangles only if there are more than 3
    if (rects.size() > 3)
        removeUniformRect(rects, mskd, 10);

    // Merge the rectangles
    if(merge)
        mergeOverlapRect(rects, 5);
    if (!merge)
        mergeOverlapRect(rects, 4050);
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
                // Apply diffusion equation
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
    //Initialize NUM_IMAGES empty vectors
    std::vector<std::vector<cv::Rect>> processedData2;

    //For each image retrieve the Rects and put them in the right vector
    for (int i = 0; i < NUM_IMAGES; i++)
    {
		std::vector<cv::Rect> processedData;

        for (int j = 0; j < bbs.size(); j++)
        {
            if (bbs[j].fileNum == i+1)
            {
				cv::Rect r = cv::Rect(bbs[j].x1, bbs[j].y1, bbs[j].width, bbs[j].height);
				processedData.push_back(r);
			}
		}

		processedData2.push_back(processedData);
	}

    return processedData2;
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

std::vector<int> classify(cv::Mat& image, std::vector<cv::Rect>& rects, bool recurse) {
    // Compute and normalize histograms for each bounding box
    std::vector<cv::Mat> histograms; // Store histograms for each box
    for (const auto& box : rects) {
        cv::Mat hist;

        // Copy the bounding box image without modifying the original image
        cv::Mat boxImage = image(box).clone();

        // Compute histogram in HSV colorspace using all channels
        cv::cvtColor(boxImage, boxImage, cv::COLOR_BGR2HSV);
        std::vector<cv::Mat> channels;
        cv::split(boxImage, channels);
        int histSize = 256;
        float range[] = { 0, 256 };
        const float* histRange = { range };
        bool uniform = true;
        bool accumulate = false;

        // Compute histogram for each channel and merge them
        std::vector<cv::Mat> channelHist;
        for (int i = 0; i < channels.size(); i++) {
            cv::Mat channelHistSingle;
            cv::calcHist(&channels[i], 1, 0, cv::Mat(), channelHistSingle, 1, &histSize, &histRange, uniform, accumulate);
            channelHist.push_back(channelHistSingle);
        }
        cv::Mat mergedHist;
        cv::merge(channelHist, mergedHist);

        // Normalize the histogram
        cv::normalize(mergedHist, hist, 0, 1, cv::NORM_MINMAX, -1, cv::Mat());

        histograms.push_back(hist);
    }

    // Compute the distance between each pair of histograms
    std::vector<std::vector<double>> distances;
    for (size_t i = 0; i < histograms.size(); i++) {
		std::vector<double> row;
        for (size_t j = 0; j < histograms.size(); j++) {
			double dist = cv::compareHist(histograms[i], histograms[j], cv::HISTCMP_CHISQR);
			row.push_back(dist);
		}
		distances.push_back(row);
	}
    // Group the bounding boxes into 2 clusters, using the distance matrix
    std::vector<int> labels;
    cv::Mat labelsMat;
    cv::Mat distancesMat(distances.size(), distances.size(), CV_32F);
    for (size_t i = 0; i < distances.size(); i++) {
        for (size_t j = 0; j < distances.size(); j++) {
            distancesMat.at<float>(i, j) = distances[i][j];
        }
    }

    cv::kmeans(distancesMat, 2, labelsMat, cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 10, 1.0), 3, cv::KMEANS_PP_CENTERS);

    // If there are a certain number of boxes, check if the classification is correct
    bool redo = false;
    if (rects.size() == 4 || rects.size() == 5 && recurse) {
        redo = class_clean(rects, labelsMat, image);
    }

    //If the classification is not correct, call the function again
    if (redo) {
       return classify(image, rects, false);
    }

    // labels must be 1 and 2
    for (int i = 0; i < labelsMat.rows; i++) {
		labels.push_back(labelsMat.at<int>(i, 0) + 1);
	}

    return labels;
}

bool class_clean(std::vector<cv::Rect>& rects, cv::Mat& labelsMat, cv::Mat& image) {
    bool redo = false;
    // Count the number of boxes for each label
    std::vector<int> labelCount(2, 0);
    for (int i = 0; i < labelsMat.rows; i++) {
        labelCount[labelsMat.at<int>(i, 0)]++;
    }
    // If one label is assigned to only one box, and the height of the box is less than 1/3 of the image heigth, delete the box
    for (int i = 0; i < labelCount.size(); i++) {
        if (labelCount[i] == 1) {
            for (int j = 0; j < labelsMat.rows; j++) {
                if (labelsMat.at<int>(j, 0) == i) {
                    if (rects[j].height < image.rows / 3) {
                        rects.erase(rects.begin() + j);
                        redo = true;
                    }
                }
            }
        }
    }
    return redo;
}
