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
        AP.push_back( computeAP(resultData, predData, classCount[i]) );
    }

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
        float iou = computeIoUSEG(resultData[i], predData[i]);
        IoU.push_back(iou);
        std::cout << "*";
    }

    //print IoU for each image
    std::cout << std::endl;
    for (int i = 0; i < IoU.size(); i++) {
		std::cout << "IoU " << i+1 << ": " << IoU[i] << std::endl;
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

float computeAP(const std::vector<BoundingBox> PredData, const std::vector<BoundingBox> ResultData, int label) {
    int true_positives = 0;
    int false_positives = 0;
    //false negatives is the number of boxes with that label in predData
    int false_negatives = std::count_if(PredData.begin(), PredData.end(), [label](BoundingBox i) {return i.id == label; });

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

    double intersection_area = static_cast<double> (x2 - x1) * (y2 - y1);
    double box1_area = static_cast<double> (box1.width * box1.height);
    double box2_area = static_cast<double> (box2.width * box2.height);

    return intersection_area / (box1_area + box2_area - intersection_area);
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
        //      for (size_t i = 0; i < predData.size(); i++) {
        //          if (predData[i].fileNum == k + 1) {
              //		pred_im.push_back(predData[i]);
              //	}
              //}

        std::cout << "Image " << k + 1 << " eval" << std::endl;
        float result_bb = processBoxPreds(result_im, result_im);
        float result_bb_rev = processBoxPreds(result_im_rev, result_im);
        std::cout << "AP: " << std::max(result_bb, result_bb_rev) << std::endl;

    }
}
