// main.cpp : Francesco Pio Monaco

#include "../CV_Project/headers/Utility.h" 
#include "../CV_Project/headers/header.h"
#include "../CV_Project/headers/court_detection.h"
#include "../CV_Project/headers/player_detection.h"
#include "../CV_Project/headers/write_results.h"

// ***STRING CONSTANTS***
const std::string partial = "/ProcessedBoxes/";
const std::string complete = "/Predictions";
const std::string mask_path = "/Masks";


////// COSE DA FARE
// 
// 
// 

// ***MAIN***
int main(int argc, char** argv)
{
    // LOAD ZONE
        //Check if the number of arguments is correct
    if (argc != 2) {
		std::cout << "Usage: " << argv[0] << " <relative_path>" << std::endl;
		return -1;
	}
        //Take the rel path from argv
    std::string rel_path = argv[1];

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
    
    std::cout << "------\nStarting image processing pipeline" << std::endl;
    // BEGIN OF THE PROCESSING PIPELINE

        //Take the raw bounding boxes
    std::vector<BoundingBox> processedData = loadBoundingBoxData(rel_path + "/ProcessedBoxes", false);
        //Reorganize the vector into a vector of vectors of Rects
    std::vector<std::vector<cv::Rect>> processedData2 = reshapeBB(processedData);

    
        // For each test image
    for (size_t k = 0; k < images.size(); k++) {

        //pick the fn of the image
         cv::String fn2 = fn[k];
         int num = extractNumber(fn2);
          
         //clean the boxes
         std::cout << "Boxes before cleaning: " << processedData2[num-1].size() << std::endl;
         cleanRectangles(processedData2[num-1], images[k]);
         std::cout << "Boxes after cleaning: " << processedData2[num-1].size() << std::endl;


        //classify the boxes
        std::vector<int> labels_BB = classify(images[k], processedData2[num - 1]);

        //Draw the boxes on the image with different colors for the 3 classes
        for (size_t i = 0; i < processedData2[num - 1].size(); i++) {
			cv::rectangle(images[k], processedData2[num - 1][i], cv::Scalar(0, 255, 0), 2);
			cv::putText(images[k], std::to_string(labels_BB[i]), cv::Point(processedData2[num - 1][i].x, processedData2[num - 1][i].y), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);
		}

        cv::imshow("Image", images[k]);
		cv::waitKey(0);

        //Write the boxes to a file
        //writeBB(images[k], processedData2[num-1], labels_BB, rel_path + complete + "/im" + std::to_string(num) + "_bb.txt");

         std::string boxes = rel_path + "Images/im" + std::to_string(num);

         //string to take the image
         std::string img_path = boxes + ".jpg";

         //string to save segmentation ground truth mask
         std::string bin = boxes + "_bin.png";

         //string to save the same mask, with a different color coding
         std::string img_save = boxes + "_color.jpg";

         //bounding boxes for player detection
         boxes = boxes + "_bb.txt";

         //Strings to save the files
         boxes = rel_path + "/Predictions/im" + std::to_string(num);
         boxes = boxes + "_bb.txt";
         std::string seg_bin_file = rel_path + complete + "/im" + std::to_string(num) + "_bin.png";
         std::string seg_color_file = rel_path + complete + "/im" + std::to_string(num) + "_color.png";

         cv::Mat  image_box = images[k].clone();


         //cv::Mat seg_image(image_box.size(), CV_8UC1);
         //player_segmentation(image_box, seg_image, boxes);

         ////cv::imshow("Image", seg_image);

         //cv::Mat mask, clustered, centroid;



         //// Eliminate boxes inside the image to have a better field detection 
         // player_elimination(image_box, mask, seg_image);
         // color_quantization(image_box, clustered, centroid);
         // cv::Mat segmentation = clustered.clone();
         // field_distinction(image_box, clustered, segmentation);
         // cv::imshow("Image", segmentation);
         // cv::waitKey(0);
         // cv::Vec2f line;
         // bool val = line_refinement(image_box, line);
         // if (val) {
         //     court_segmentation_refinement(segmentation, line);
         // }

         // unifySegmentation(segmentation, seg_image, processedData2[num - 1], labels_BB);

         // cv::Mat segmentation_bin = cv::Mat::zeros(segmentation.rows, segmentation.cols, CV_8UC1);
         // createSegmentationPNG(segmentation, segmentation_bin);
         // 
         // writeSEG(segmentation_bin, seg_bin_file);
         // writeSEG(segmentation, seg_color_file);

          //cv::imshow("Final", segmentation);
          //cv::imshow("Final_bin", segmentation_bin);
          //cv::waitKey(0);
          std::cout << "*";

   ////      //for each box draw it on the image using different color for the labels
   //      for (size_t i = 0; i < processedData2[num-1].size(); i++) {
			// cv::rectangle(images[k], processedData2[num-1][i], cv::Scalar(0, 255, 0), 2);
			// cv::putText(images[k], std::to_string(labels[i]), cv::Point(processedData2[num-1][i].x, processedData2[num-1][i].y), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 2);
		 //}
    }
   

    // EVALUATION PIPELINE
    //std::cout << "------\nEvaluation Pipeline" << std::endl;
    //    // Bounding Box Evaluation
    //std::vector<BoundingBox> resultData = loadBoundingBoxData(rel_path + mask_path);
    //std::vector<BoundingBox> resultData_rev = loadBoundingBoxData(rel_path + mask_path, true, true);
    //std::vector<BoundingBox> predData = loadBoundingBoxData(rel_path + complete);
    //float result_bb = processBoxPreds(resultData, predData);
    //float result_bb_rev = processBoxPreds(resultData_rev, predData);

    //    //AP for each image
    //singleImageAP(resultData, resultData_rev, predData, images.size());


    //    // Semantic Segmentation Evaluation
    //std::vector<cv::Mat> segmentationGOLD = loadSemanticSegmentationData(rel_path + mask_path);
    //std::vector<cv::Mat> segmentationGOLD_REV = loadSemanticSegmentationData(rel_path + mask_path, true);
    //std::vector<cv::Mat> segmentationPRED = loadSemanticSegmentationData(rel_path + complete);
    //std::cout << "Semantic Segmentation Eval" << std::endl;
    //float result_seg = processSemanticSegmentation(segmentationGOLD, segmentationPRED);
    //std::cout << "Semantic Segmentation Reverse Eval" << std::endl;
    //float result_seg_rev = processSemanticSegmentation(segmentationGOLD_REV, segmentationPRED);

    //std::cout << "------\n";
    //std::cout << "mAP: " << std::max(result_bb, result_bb_rev) << std::endl;
    //std::cout << "IoU: " << std::max(result_seg, result_seg_rev) << std::endl;

        ////Show results, uncomment to show
    //showResults("D:/Download/Sport_scene_dataset/Images", "D:/Download/Sport_scene_dataset/Masks");
    
}