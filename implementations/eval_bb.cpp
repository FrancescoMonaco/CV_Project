#include "../headers/Eval.h"
#include <iostream>
#include <fstream>
#include <filesystem>
#include <math.h>
#include <numeric>
#include <iomanip>
#include <opencv2/opencv.hpp>
// Eval.cpp : Francesco Pio Monaco

//***Variables, namespaces definitions***
const float IOU_THRESH = 0.5;
const int NUM_CLASSES = 4;
const std::vector<std::string> classes = { "background", "1",\
 "2", "field"};

namespace fs = std::filesystem;


//***Functions implementations***

std::vector<BoundingBox> loadBoundingBoxData(const std::string& filePath, bool hasID, bool reverse) {
    std::vector<BoundingBox> data;
    // Go into filePath and for each .txt file
    for(const auto& entry : fs::directory_iterator(filePath)) {
        if (entry.path().extension() == ".txt") {
            // Open the file
            std::ifstream file(entry.path());
            //Convert the string to a cv string and call ectractNumber
            cv::String pt(entry.path().string());
            int fileNum = extractNumber(pt);
            // Read the file line by line
            std::string line;
            while (std::getline(file, line)) {
                // Split the line by spaces
                std::istringstream iss(line);
                std::vector<std::string> tokens{ std::istream_iterator<std::string>{iss}, std::istream_iterator<std::string>{} };
                // Create a BoundingBox object and fill it with x1 x2 width height id
                BoundingBox bb;
                if(hasID and !reverse)
                    bb.id = std::stoi(tokens[4]);
                if (hasID and reverse) {
                    bb.id = std::stoi(tokens[4]);
                    bb.id = (bb.id == 1) ? 2 : 1;
                }
                bb.x1 = std::stoi(tokens[0]);
                bb.y1 = std::stoi(tokens[1]);
                bb.width = std::stoi(tokens[2]);
                bb.height = std::stoi(tokens[3]);
                bb.fileNum = fileNum;

                // Add the BoundingBox object to the vector
                data.push_back(bb);
            }
        }
    }
    return data;
}

float processBoxPreds(const std::vector<BoundingBox>& resultData, const std::vector<BoundingBox>& predData) {
    //go into resultData and count the different classes looking at the id
    int numClasses = 0;
    std::vector<int> classCount;
    for (const BoundingBox& bb : resultData) {
        if (std::find(classCount.begin(), classCount.end(), bb.id) == classCount.end()) {
			classCount.push_back(bb.id);
			numClasses++;
		}
	}
    //for each class compute AP and push it into a vector
    std::vector<float> AP;

    for (int i = 0; i < numClasses; i++) {
        AP.push_back( computeAP(predData, resultData, classCount[i]) );
    }
    // Compute mAP
    float mAP = 0.0f;
    for (float value : AP) {
        mAP += value;
    }
    mAP = mAP / AP.size();
    return mAP;
}

float computeAP(const std::vector<BoundingBox> PredData, const std::vector<BoundingBox> ResultData, int label) {
    int true_positives = 0;
    int false_positives = 0;
    //false negatives is the number of boxes with that label in ResultData (every box should be matched)
    int false_negatives = std::count_if(ResultData.begin(), ResultData.end(), [label](BoundingBox i) {return i.id == label; });

    for (const BoundingBox& prediction : ResultData) {
        if (prediction.id != label) {
            continue;
        }

        bool is_true_positive = false;

        for (const BoundingBox& gt_box : PredData) {
            if (gt_box.id == label && computeIoUBB(prediction, gt_box) >= IOU_THRESH) {
                is_true_positive = true;
                break;
            }
        }

        if (is_true_positive) {
            // If the prediction is a true positive, remove it from the false negatives
            true_positives++;
            false_negatives--;
        }
        else {
            false_positives++;
        }
    }

    if (true_positives == 0 && false_positives == 0) {
        return 0.0;  // Avoid division by zero
    }

    float precision = static_cast<float>(true_positives) / (true_positives + false_positives);
    float recall = static_cast<float>(true_positives) / (true_positives + false_negatives);

    return precision * recall;
}

double computeIoUBB(const BoundingBox& box1, const BoundingBox& box2) {
    int x1 = std::max(box1.x1, box2.x1);
    int y1 = std::max(box1.y1, box2.y1);
    int x2 = std::min(box1.x1 + box1.width, box2.x1 + box2.width);
    int y2 = std::min(box1.y1 + box1.height, box2.y1 + box2.height);

    if (x1 >= x2 || y1 >= y2) {
        return 0.0;  // No overlap
    }

    double intersection_area = ((static_cast<double> (x2) - x1)) * (static_cast<double> (y2) - y1);
    double box1_area = static_cast<double> (box1.width) * (box1.height);
    double box2_area = static_cast<double> (box2.width) * (box2.height);

    return intersection_area / (box1_area + box2_area - intersection_area);
}

void singleImageAP(const std::vector<BoundingBox> resultData, const std::vector<BoundingBox> resultData_rev ,const std::vector<BoundingBox> predData,int size) {
    for (size_t k = 0; k < size; k++) {
        std::vector<BoundingBox> result_im, result_im_rev, pred_im;
        //go into the vectors resultData and predData and extract the bounding boxes for the current image, look into fileNum variable
        for (size_t i = 0; i < resultData.size(); i++) {
            if (resultData[i].fileNum == k + 1) {
                result_im.push_back(resultData[i]);
            }
        }
        for (size_t i = 0; i < resultData_rev.size(); i++) {
            if (resultData_rev[i].fileNum == k + 1) {
                result_im_rev.push_back(resultData_rev[i]);
            }
        }
        for (size_t i = 0; i < predData.size(); i++) {
            if (predData[i].fileNum == k + 1) {
              		pred_im.push_back(predData[i]);
            }
        }

        std::cout << "Image " << k + 1 << " eval" << std::endl;
        float result_bb = processBoxPreds(result_im, pred_im);
        float result_bb_rev = processBoxPreds(result_im_rev, pred_im);
        std::cout << "AP: " << std::max(result_bb, result_bb_rev) << std::endl;

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