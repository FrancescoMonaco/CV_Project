// CV_Project.cpp : Francesco Pio Monaco

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>

#include <string>
#include <iostream>
#include <vector>

#include "../CV_Project/headers/Utility.h" 

//Constants, later to be moved to a config file
const std::string num_path = "D:/Download/process_data/Rec_templates";
const std::string part_path = "D:/Download/process_data/Body_templates";
const std::string rel_path = "D:/Download/Sport_scene_dataset";

const double color_variation_threshold = 10.0;
const double hu_moments_threshold = 0.46;
int orange_lower_bound = 120;
int orange_upper_bound = 255;

int main()
{

    std::string path = rel_path + "/Images/*.jpg"; //select only jpg

    // Load images
    std::vector<cv::Mat> images;
    std::vector<cv::String> fn;
    cv::glob(path, fn, true);

    for (size_t k = 0; k < fn.size(); ++k)
    {
        cv::Mat im = cv::imread(fn[k]);
        if (im.empty()) continue; //only proceed if successful
        images.push_back(im);
    }

    // BEGIN OF THE PROCESSING PIPELINE
    int savenum= 0;
    for (auto& test : images)
    {
        //Create a copy of the image to work on
        cv::Mat test_copy = test.clone();

    }
    std::vector<BoundingBox> processedData = loadBoundingBoxData(rel_path + "/ProcessedBoxes", false);
    //Reorganize the vector into a vector of vectors of BoundingBoxes
    std::vector<std::vector<cv::Rect>> processedData2 = reshapeBB(processedData);

    // for each image, keep also the relative fn during the loop
    for (size_t k = 0; k < images.size(); k++) {
        //pick the fn of the image
         cv::String fn2 = fn[k];
         int num = extractNumber(fn2);
         std::cout << num << std::endl;
         std::cout << "Boxes before cleaning: " << processedData2[num-1].size() << std::endl;
         cleanRectangles(processedData2[num-1], images[k]);
         std::cout << "Boxes after cleaning: " << processedData2[num-1].size() << std::endl;
         //show the image with the boxes
         for (auto& r : processedData2[num - 1]) {
			 cv::rectangle(images[k], r, cv::Scalar(0, 255, 0), 2);
		 }
         cv::imshow("Image", images[k]);
		 cv::waitKey(0);
    }


    /*
    //Evaluation Part
    //std::vector<BoundingBox> resultData = loadBoundingBoxData("D:/Download/Sport_scene_dataset/Masks");
    //std::vector<BoundingBox> predData = loadBoundingBoxData("D:/Download/Sport_scene_dataset/Masks");
    float result = processBoxPreds(resultData, resultData);
    std::cout << "mAP: " << result << std::endl;

    std::vector<cv::Mat> resultData2 = loadSemanticSegmentationData("D:/Download/Sport_scene_dataset/Masks");
    float result2 = processSemanticSegmentation(resultData2, resultData2);
    std::cout << "IoU: " << result2 << std::endl;

    //Show results, uncomment to show
    showResults("D:/Download/Sport_scene_dataset/Images", "D:/Download/Sport_scene_dataset/Masks");
    */
}