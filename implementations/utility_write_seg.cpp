#include "../headers/Utility.h"
#include <fstream>

// Utility_Write_Seg.cpp : Francesco Pio Monaco
// 
//***Constants for the utility functions
//Colors for the segmentation
std::vector<cv::Vec3b> colors = { cv::Vec3b(0, 0, 255), cv::Vec3b(255, 0, 0) };

//***Implementations

void writeBB(cv::Mat& image, std::vector<cv::Rect> rects, std::vector<int> labels, std::string rel_path)
{
    std::ofstream file(rel_path);

    if (!file.is_open()) {
        std::cerr << "Error opening file: " << rel_path << std::endl;
        return;
    }

    // Write the bounding boxes
    for (size_t i = 0; i < rects.size(); ++i) {
        cv::Rect rect = rects[i];
        int label = labels[i];

        // Write the data in the format x y width height label
        file << rect.x << " " << rect.y << " " << rect.width << " " << rect.height << " " << label << "\n";
    }

    file.close();
}

void unifySegmentation(cv::Mat& segmentation, cv::Mat& players, std::vector<cv::Rect> rects, std::vector<int> labels)
{
    //For each box go into seg_image take all the 1s and color in segmentation with the label
    for (size_t i = 0; i < rects.size(); i++) {
        cv::Mat mask = players(rects[i]);
        for (int j = 0; j < mask.rows; j++) {
            for (int k = 0; k < mask.cols; k++) {
                //print the value of the mask
                if (mask.at<uchar>(j, k) == 255) {
                    //write the label in the segmentation image moving in the right position
                    segmentation.at<cv::Vec3b>(j + rects[i].y, k + rects[i].x) = colors[labels[i] - 1];

                }
            }
        }
    }
}

void createSegmentationPNG(cv::Mat segmentation, cv::Mat& gray_seg)
{
    //for each pixel in the segmentation image, if black put 0; if green put 3, if red put 1, if blue put 2
    for (int i = 0; i < segmentation.rows; i++) {
        for (int j = 0; j < segmentation.cols; j++) {
            if (segmentation.at<cv::Vec3b>(i, j) == cv::Vec3b(0, 0, 0)) {
                gray_seg.at<uchar>(i, j) = 0;
            }
            else if (segmentation.at<cv::Vec3b>(i, j) == cv::Vec3b(0, 255, 0)) {
                gray_seg.at<uchar>(i, j) = 3;
            }
            else if (segmentation.at<cv::Vec3b>(i, j) == cv::Vec3b(255, 0, 0)) {
                gray_seg.at<uchar>(i, j) = 1;
            }
            else if (segmentation.at<cv::Vec3b>(i, j) == cv::Vec3b(0, 0, 255)) {
                gray_seg.at<uchar>(i, j) = 2;
            }
        }
    }
}

void writeSEG(cv::Mat& image, std::string rel_path)
{
    //save the image in the path
    cv::imwrite(rel_path, image);
}

