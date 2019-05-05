/*
 * Bebop2Demo.cpp
 *
 *  Created on: Feb 1, 2019
 *      Author: slascos
 */
#include "OpenCVProcessing.h"
#include <ctime>

using namespace std;
using namespace cv;
#include "opencv2/calib3d.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
//#include "opencv2/core/cuda.hpp"
//#include "opencv2/cudaimgproc.hpp"

#include <iostream>
#include <fstream>

using namespace cv;
using namespace cv::dnn;
using namespace cv::xfeatures2d;

void harrisCorner(Mat &grayImage, Mat &outputImage)
{
    Mat corners, cornersNorm, cornersNormScaled;
    int thresh = 100;

    cornerHarris(grayImage, corners, 7, 5, 0.05, BORDER_DEFAULT);
    normalize(corners, cornersNorm, 0, 255, NORM_MINMAX, CV_32FC1, Mat());
    convertScaleAbs(cornersNorm, cornersNormScaled);

    // Draw circles around corners
    for (int j = 0; j < cornersNorm.rows; j++) {
        for (int i = 0; i < cornersNorm.cols; i++) {
            if ( (int) cornersNorm.at<float>(j,i) > thresh ) {
                circle (cornersNormScaled, Point(i,j), 5, Scalar(255), 2, 8, 0);
            }
        }
    }
    cornersNormScaled.copyTo(outputImage);
}


void openCVProcessing(shared_ptr<Mat> imageToProcess, bool *processingDone)
{
    //Capture a temporary image from the camera
    Mat imgOriginal;
    cvtColor(*imageToProcess, imgOriginal, COLOR_BGR2HSV); // Convert to HSV
    
    // Time
    time_t tt = time(NULL);
    string time_s = ctime(&tt);
    time_s = "Time: " + time_s.substr(0, time_s.size()-1);

    // Save normal image every x number of times
    //if (counter % 10 == 0)
    //    imwrite("normal: " + s + ".jpg", imgOriginal);

    // Magenta
    int mLowH = 317/2;
    int mHighH = 345/2;

    int mLowS = 50; 
    int mHighS = 200;

    int mLowV = 50;
    int mHighV = 255;

    // 317 - 345
    // 27 - 65
    // 62 - 100

    // Green
    int gLowH = 72/2;
    int gHighH = 97/2;

    int gLowS = 50; 
    int gHighS = 200;

    int gLowV = 50;
    int gHighV = 255;

    // 72 - 97
    // 38 - 83
    // 55 - 80

    //Create a black image with the size as the camera output
    Mat imgLines = Mat::zeros(imgOriginal.size(), CV_8UC3);;

    //Threshold the image
    Mat imgThresholded1;
    inRange(imgOriginal, Scalar(mLowH, mLowS, mLowV), Scalar(mHighH, mHighS, mHighV), imgThresholded1);
    Mat imgThresholded2;
    inRange(imgOriginal, Scalar(gLowH, gLowS, gLowV), Scalar(gHighH, gHighS, gHighV), imgThresholded2);  

    Mat imgThresholded;
    imgThresholded = imgThresholded1 + imgThresholded2; 

    //morphological opening (removes small objects from the foreground)
    erode(imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)) );
    dilate(imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)) ); 

    //morphological closing (removes small holes from the foreground)
    dilate(imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)) ); 
    erode(imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)) );

    //Calculate the moments of the thresholded image
    Moments oMoments = moments(imgThresholded);

    double dM01 = oMoments.m01;
    double dM10 = oMoments.m10;
    double dArea = oMoments.m00;

    cvtColor(imgThresholded, imgThresholded, COLOR_GRAY2BGR);

    // if the area <= 10000, I consider that the there are no object in the image and it's because of the noise, the area is not zero 
    if (dArea > 10000)
    {
      //calculate the position of the ball
      int posX = dM10 / dArea;
      int posY = dM01 / dArea;

      //*alpha_x = posX
      //*alpha_y = posY

      string position_s = "Position: " + to_string(posX) + ", " + to_string(posY);
      putText(imgThresholded, position_s, Point(0, 125), FONT_HERSHEY_PLAIN, 3.0, Scalar(255, 255, 255), 2, 8, false);
      circle(imgThresholded, Point(posX, posY), 30, Scalar(0,0,255), 10, 8, 0);

    }

    putText(imgThresholded, time_s, Point(0, 25), FONT_HERSHEY_PLAIN, 3.0, Scalar(255, 255, 255), 2, 8, false);
    
    ////

    // Output Image
    imgThresholded.copyTo(*imageToProcess);

    *processingDone = true;
}
