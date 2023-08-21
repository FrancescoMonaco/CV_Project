// Eval.cpp : Francesco Pio Monaco
#include "../headers/Eval.h"
#include <iostream>
#include <fstream>
#include <filesystem>
#include <math.h>
#include <numeric>
#include <iomanip>
#include <opencv2/opencv.hpp>

//***Variables, namespaces definitions***
const float IOU_THRESH = 0.5;
const int NUM_CLASSES = 4;
const std::vector<std::string> classes = { "background", "1",\
 "2", "field"};

std::vector<cv::Vec3b> customColormap = {
    cv::Vec3b(0, 0, 0),       // Black
    cv::Vec3b(0, 0, 255),     // Green
    cv::Vec3b(255, 0, 0),     // Blue
    cv::Vec3b(0, 255, 0),   // Yellow
};

namespace fs = std::filesystem;


//***Functions implementations***

std::vector<BoundingBox> loadBoundingBoxData(const std::string& filePath, bool hasID) {
    std::vector<BoundingBox> data;
    int fileNum = 0;
    // Go into filePath and for each .txt file
    for(const auto& entry : fs::directory_iterator(filePath)) {
        if (entry.path().extension() == ".txt") {
            // Open the file
            std::ifstream file(entry.path());
            // Read the file line by line
            std::string line;
            while (std::getline(file, line)) {
                // Split the line by spaces
                std::istringstream iss(line);
                std::vector<std::string> tokens{ std::istream_iterator<std::string>{iss}, std::istream_iterator<std::string>{} };
                // Create a BoundingBox object and fill it with x1 x2 width height id
                BoundingBox bb;
                if(hasID)
                    bb.id = std::stoi(tokens[4]);
                bb.x1 = std::stoi(tokens[0]);
                bb.y1 = std::stoi(tokens[1]);
                bb.width = std::stoi(tokens[2]);
                bb.height = std::stoi(tokens[3]);
                bb.fileNum = fileNum;

                // Add the BoundingBox object to the vector
                data.push_back(bb);
            }
            fileNum++;
        }
    }
    return data;
}

std::vector<cv::Mat> loadSemanticSegmentationData(const std::string& filePath) {
    //for each .png file that has _bin in the name in filePath push back the image in the vector
    std::vector<cv::Mat> data;
    for (const auto& entry : fs::directory_iterator(filePath)) {
        if (entry.path().extension() == ".png" && entry.path().filename().string().find("_bin") != std::string::npos) {
            data.push_back(cv::imread(entry.path().string(), cv::IMREAD_GRAYSCALE));
        }
    }
    return data;
}

float processBoxPreds(const std::vector<BoundingBox>& resultData, const std::vector<BoundingBox>& predData) {
    // Confusion matrix data
    std::vector<int> TP(2, 0);
    std::vector<int> FP(2, 0);
    std::vector<int> FN(2, 0);
    // Recall, precision vectors, updated at each iteration for later use in the average precision calculation
    std::vector<std::vector<float>> recall(2, std::vector<float>(0, 0));
    std::vector<std::vector<float>> precision(2, std::vector<float>(0, 0));

    // For each file in both resultData check in predData
    for (int i = 0; i < resultData.size(); i++) {

        //if in predData there's no file with same name and same id, then it's a false negative
        bool found = false;
        for (int j = 0; j < predData.size(); j++) {
            if (resultData[i].id == predData[j].id && resultData[i].fileNum == predData[j].fileNum) {
                found = true;
                break;
            }
        }
        if (!found) {
            FN[resultData[i].id-1]++;
            // Compute precision and recall at the time point and update the vectors
            precision[resultData[i].id-1].push_back((float)TP[resultData[i].id-1] / (TP[resultData[i].id-1] + FP[resultData[i].id-1]));
            recall[resultData[i].id-1].push_back((float)TP[resultData[i].id-1] / (TP[resultData[i].id-1] + FN[resultData[i].id-1]));
            continue;
        }

        std::vector<float> ious;
        // Compare resultData[i] with all the preddata with the same fileNum and id and pick the one with the highest IOU
        for (int j = 0; j < predData.size(); j++) {


            if (resultData[i].id == predData[j].id && resultData[i].fileNum == predData[j].fileNum) {

                int xA = std::max(resultData[i].x1, predData[j].x1);
                int yA = std::max(resultData[i].y1, predData[j].y1);
                int xB = std::min(resultData[i].x1 + resultData[i].width, predData[j].x1 + predData[j].width);
                int yB = std::min(resultData[i].y1 + resultData[i].height, predData[j].y1 + predData[j].height);

                float interArea = std::max(0, xB - xA + 1) * std::max(0, yB - yA + 1);
                
                float boxAArea = resultData[i].width * resultData[i].height;
                float boxBArea = predData[j].width * predData[j].height;

                float iou = interArea / (boxAArea + boxBArea - interArea);

                ious.push_back(iou);
            }
        }

        // If the highest IOU is greater than the threshold, it's a true positive
        if (ious.size() > 0 && *std::max_element(ious.begin(), ious.end()) > IOU_THRESH) {
            TP[resultData[i].id-1]++;
            // Compute precision and recall at the time point and update the vectors
            precision[resultData[i].id-1].push_back((float)TP[resultData[i].id-1] / (TP[resultData[i].id-1] + FP[resultData[i].id-1]));
            recall[resultData[i].id-1].push_back((float)TP[resultData[i].id-1] / (TP[resultData[i].id-1] + FN[resultData[i].id-1]));
        }
        else {
            FP[resultData[i].id]++;
            // Compute precision and recall at the time point and update the vectors
            precision[resultData[i].id-1].push_back((float)TP[resultData[i].id-1] / (TP[resultData[i].id-1] + FP[resultData[i].id-1]));
            recall[resultData[i].id-1].push_back((float)TP[resultData[i].id-1] / (TP[resultData[i].id-1] + FN[resultData[i].id-1]));
        }

        //Every 4 iterations print a *
        if (i % 4 == 0) {
			std::cout << "*";
		}
    }
    std::cout << std::endl;
    // Compute average precision for class
    std::vector<float> AP = computeAP(recall, precision);

    // Compute mAP
    float mAP = 0.0f;
    for (float value : AP) {
        mAP += value;
    }
    mAP = mAP / AP.size();
    return mAP;
}

float processSemanticSegmentation(const std::vector<cv::Mat>& resultData, const std::vector<cv::Mat>& predData)
{
    //for each couple of allineated images, compute IoU and push it into a vector
    std::vector<float> IoU;
    for (int i = 0; i < resultData.size(); i++) {
        float iou = computeIoU(resultData[i], predData[i]);
        IoU.push_back(iou);
        std::cout << "*";
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

float computeIoU(const cv::Mat& result, const cv::Mat& pred)
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

std::vector<float> computeAP(const std::vector<std::vector<float>>& precision, const std::vector<std::vector<float>>& recall) {
    std::vector<float> ap;
    // we start from 1 to skip the background class
    for (size_t i = 0; i < precision.size(); ++i) {
        const std::vector<float>& prec = precision[i];
        const std::vector<float>& rec = recall[i];

        // Sort precision and recall vectors in descending order of recall
        std::vector<size_t> indices(prec.size());
        std::iota(indices.begin(), indices.end(), 0);
        std::sort(indices.begin(), indices.end(), [&rec](size_t a, size_t b) { return rec[a] > rec[b]; });

        // Compute interpolated precision, 
        std::vector<float> interpolatedPrec(indices.size());
        float maxPrec = 0.0f;
        for (size_t j = 0; j < indices.size(); ++j) {
            size_t idx = indices[j];
            maxPrec = std::max(maxPrec, prec[idx]);
            interpolatedPrec[j] = maxPrec;
        }

        // Compute average precision
        float sumPrec = std::accumulate(interpolatedPrec.begin(), interpolatedPrec.end(), 0.0f);
        ap.push_back(sumPrec / static_cast<float>(interpolatedPrec.size()));
    }

    return ap;
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

int extractNumber(const cv::String& s) {
    size_t start = s.find_first_of("0123456789");
    if (start != cv::String::npos) {
        size_t end = s.find_first_not_of("0123456789", start);
        if (end != cv::String::npos) {
            return std::stoi(s.substr(start, end - start));
        }
    }
    return -1; // Return a sentinel value indicating failure
}
