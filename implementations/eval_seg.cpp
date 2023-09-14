#include "../headers/Eval.h"
#include <iostream>
#include <fstream>
#include <filesystem>
#include <math.h>
#include <numeric>
#include <iomanip>
#include <opencv2/opencv.hpp>

// eval_seg : Francesco Pio Monaco

//***Variables, namespaces definitions***
const float IOU_THRESH = 0.5;
const int NUM_CLASSES = 4;
const std::vector<std::string> classes = { "background", "1",\
 "2", "field" };

std::vector<cv::Vec3b> customColormap = {
    cv::Vec3b(0, 0, 0),       // Black
    cv::Vec3b(0, 0, 255),     // Green
    cv::Vec3b(255, 0, 0),     // Blue
    cv::Vec3b(0, 255, 0),   // Yellow
};

namespace fs = std::filesystem;


//***Functions implementations***
std::vector<cv::Mat> loadSemanticSegmentationData(const std::string& filePath, bool reverse) {
    //for each .png file that has _bin in the name in filePath push back the image in the vector
    std::vector<cv::Mat> data;
    for (const auto& entry : fs::directory_iterator(filePath)) {
        if (entry.path().extension() == ".png" && entry.path().filename().string().find("_bin") != std::string::npos) {
            data.push_back(cv::imread(entry.path().string(), cv::IMREAD_GRAYSCALE));
        }
        // if reverse then 1 and 2 are swapped
        if (reverse) {
            for (cv::Mat& image : data) {
                cv::MatIterator_<uchar> it, end;
                for (it = image.begin<uchar>(), end = image.end<uchar>(); it != end; ++it) {
                    if (*it == 1)
                        *it = 2;
                    else if (*it == 2)
                        *it = 1;
                }
            }
        }
    }
    return data;
}

float computeIoUSEG(const cv::Mat& result, const cv::Mat& pred)
{
    //compute intersection and union for each class
    std::vector<float> intersection(NUM_CLASSES, 0.0f);
    std::vector<float> unionArea(NUM_CLASSES, 0.0f);
    for (int i = 0; i < result.rows; i++) {
        for (int j = 0; j < result.cols; j++) {
            if (result.at<uchar>(i, j) == pred.at<uchar>(i, j)) {
                intersection[result.at<uchar>(i, j)]++;
            }
            unionArea[result.at<uchar>(i, j)]++;
            unionArea[pred.at<uchar>(i, j)]++;
        }
    }
    // union is the sum of the two areas minus the intersection
    for (int i = 0; i < NUM_CLASSES; i++) {
        unionArea[i] -= intersection[i];
    }
    // compute IoU for each class
    std::vector<float> IoU(NUM_CLASSES, 0.0f);
    for (int i = 0; i < NUM_CLASSES; i++) {
        //don't compute if for classes that have 0 in result
        if (unionArea[i] == 0) {
            continue;
        }
        IoU[i] = intersection[i] / unionArea[i];
    }
    // return mean IoU
    float mIoU = 0.0f;
    for (float value : IoU) {
        mIoU += value;
    }
    // divide by number of classes that do not have 0 in IoU
    int count = std::count_if(IoU.begin(), IoU.end(), [](float i) {return i > 0.0f; });
    mIoU = mIoU / count;
    return mIoU;
}

float processSemanticSegmentation(const std::vector<cv::Mat>& resultData, const std::vector<cv::Mat>& predData)
{
    //for each couple of allineated images, compute IoU and push it into a vector
    std::vector<float> IoU;
    for (int i = 0; i < resultData.size(); i++) {
        float iou = computeIoUSEG(resultData[i], predData[i]);
        IoU.push_back(iou);
        std::cout << "*";
    }

    //print IoU for each image
    std::cout << std::endl;
    for (int i = 0; i < IoU.size(); i++) {
        std::cout << "IoU " << i + 1 << ": " << IoU[i] << std::endl;
    }

    // return mean IoU
    float mIoU = 0.0f;
    for (float value : IoU) {
        mIoU += value;
    }
    mIoU = mIoU / IoU.size();
    std::cout << std::endl;
    return mIoU;
}

void showResults(const std::string& source, const std::string& path)
{
    //save in a vector all the png in the folder that have _bin in the name
    std::vector<cv::String> fn;
    std::vector<cv::String> fn_source;
    cv::glob(path + "/*_bin.png", fn, true);
    cv::glob(source + "/*.jpg", fn_source, true);

    //sort both vectors otherwise the images will not be aligned
    std::sort(fn.begin(), fn.end(), [](const cv::String& a, const cv::String& b) {
        return extractNumber(a) < extractNumber(b);
        });

    std::sort(fn_source.begin(), fn_source.end(), [](const cv::String& a, const cv::String& b) {
        return extractNumber(a) < extractNumber(b);
        });



    //Put all source images in a vector
    std::vector<cv::Mat> sourceImages;
    for (size_t k = 0; k < fn_source.size(); ++k) {
        cv::Mat im = cv::imread(fn_source[k]);
        //If the image is empty, throw an error
        if (im.empty()) {
            std::cout << "Error opening image" << std::endl;
            exit(EXIT_FAILURE);
        }
        sourceImages.push_back(im);
    }

    //for each image show the corresponding bounding box
    for (size_t k = 0; k < fn_source.size(); ++k) {
        cv::Mat im = sourceImages[k];
        //read the corresponding txt, remove _bin.png and add _bb.txt
        std::string txtPath = fn[k].substr(0, fn[k].size() - 8) + "_bb.txt";

        std::ifstream file(txtPath);
        //for each line in the txt
        std::string line;
        while (std::getline(file, line)) {

            //split the line
            std::istringstream iss(line);
            std::vector<std::string> results((std::istream_iterator<std::string>(iss)),
                std::istream_iterator<std::string>());

            //get the class and the coordinates, a line is like x y width height class
            int x = std::stoi(results[0]);
            int y = std::stoi(results[1]);
            int w = std::stoi(results[2]);
            int h = std::stoi(results[3]);
            int classId = std::stoi(results[4]);

            //draw the bounding box,different color for each class
            cv::Scalar color;
            switch (classId) {
            case 1:
                color = cv::Scalar(0, 0, 255);
                break;
            case 2:
                color = cv::Scalar(0, 255, 0);
                break;
            }
            cv::rectangle(im, cv::Rect(x, y, w, h), color, 2);

            //put the class name
            std::string className = classes[classId];
            int baseLine;
            cv::Size labelSize = cv::getTextSize(className, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
            cv::putText(im, className, cv::Point(x, y + labelSize.height), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                color);
        }

        //show the image with the bounding boxes
        cv::namedWindow("Bounding box image", cv::WINDOW_GUI_NORMAL);
        cv::imshow("Bounding box image", im);
        cv::waitKey(0);
    }


    for (size_t k = 0; k < fn.size(); ++k) {
        cv::Mat im = cv::imread(fn[k], cv::IMREAD_GRAYSCALE);
        //Show the regions colored in overlay with the original image
        cv::Mat im_color(im.size(), CV_8UC3);
        for (int i = 0; i < im.rows; i++) {
            for (int j = 0; j < im.cols; j++) {
                int index = im.at<uchar>(i, j);
                im_color.at<cv::Vec3b>(i, j) = customColormap[index];
            }
        }

        double alpha = 0.1;
        cv::Mat blended;
        cv::Mat original_image = sourceImages[k];
        cv::addWeighted(original_image, alpha, im_color, 1 - alpha, 0, blended);
        cv::namedWindow("Segmentation Overlay", cv::WINDOW_GUI_NORMAL);
        cv::imshow("Segmentation Overlay", blended);

        cv::waitKey(0);
    }

}