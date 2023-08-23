#include "header.h"
#include "court_detection.h"
#include "player_detection.h"
#include "write_results.h"


const double color_variation_threshold = 10.0;
const double hu_moments_threshold = 0.46;
int orange_lower_bound = 120;
int orange_upper_bound = 255;

int main()
{
    //Put all images in a vector using glob
    std::vector<cv::Mat> images;
    std::vector<cv::Mat> numbers;
    std::vector<cv::Mat> parts;

    std::string abs_path = "C:/Users/miche/OneDrive/Desktop/Sport_scene_dataset/Images/im";

    std::string path = "C:/Users/miche/OneDrive/Desktop/Sport_scene_dataset/Images/*.jpg"; //select only jpg
    std::vector<cv::String> fn;
    std::vector<cv::String> fn2;
    std::vector<cv::String> fn3;
    cv::glob(path, fn, true); // recurse
    //  cv::glob(num_path, fn2, true); // recurse
      //cv::glob(part_path, fn3, true); // recurse

    for (size_t k = 0; k < fn.size(); ++k)
    {
        cv::Mat im = cv::imread(fn[k]);
        if (im.empty()) continue; //only proceed if successful
        images.push_back(im);
    }

    for (size_t k = 0; k < fn2.size(); ++k)
    {
        cv::Mat im = cv::imread(fn2[k], cv::IMREAD_GRAYSCALE);
        if (im.empty()) continue; //only proceed if successful
        numbers.push_back(im);

    }

    for (size_t k = 0; k < fn3.size(); ++k) {
        cv::Mat im = cv::imread(fn3[k], cv::IMREAD_GRAYSCALE);
        if (im.empty()) continue;
        parts.push_back(im);
    }

    int i = 1;


    for (int num = 0; num < images.size(); num++)
    {
        std::string boxes = abs_path + std::to_string(i);
        
        //string to take the image
        std::string img_path = boxes + ".jpg";
        
        //string to save segmentation ground truth mask
        std::string bin = boxes + "_bin.png";
        
        //string to save the same mask, with a different color coding
        std::string img_save = boxes + "_color.jpg";
   
        //bounding boxes for player detection
        boxes = boxes + "_bb.txt";

        //test settings
        boxes = "C:/Users/miche/OneDrive/Desktop/Sport_scene_dataset/Masks/im"+ std::to_string(i);
        boxes = boxes+ "_bb.txt";

        //Create a copy of the image to work on
        cv::Mat test = cv::imread(img_path);

        cv::Mat  image_box = test.clone();


        //eliminate boxes inside the image to have a better field detection 
        box_elimination(image_box, image_box, boxes);

        //cover the holes left by the remotion of the boxes
        fill_image(image_box);

        //clusterize the image in order to detect the field and the background  
        cv::Mat clustered;
        color_quantization(image_box, clustered);

        //assign specific label for the test task
        cv::Mat segmented_field = clustered.clone();
        field_distinction(clustered, segmented_field);


        //start players segmentation
        //player_segmentation(test, image_box, boxes);

        //unire cose di player segmentation and field 

        //save bin
       // cv::Mat save_bin (segmented_field.rows, segmented_field.cols,CV_8UC1);
       // write_segmentation_results(segmented_field,save_bin,bin);

        i++;
    }
}