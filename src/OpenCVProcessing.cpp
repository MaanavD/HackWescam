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


void colourThresholding(std::shared_ptr<cv::Mat> imageToProcess, bool *processingDone, int *alpha_x, int *alpha_y)
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
    int mHighS = 255;

    int mLowV = 50;
    int mHighV = 255;

    // 317 - 345
    // 27 - 65
    // 62 - 100

    // Green
    int gLowH = 72/2;
    int gHighH = 97/2;

    int gLowS = 50; 
    int gHighS = 255;

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
    erode(imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(4, 4)) );
    dilate(imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(4, 4)) ); 

    //morphological closing (removes small holes from the foreground)
    dilate(imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(4, 4)) ); 
    erode(imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(4, 4)) );

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

      *alpha_x = posX;
      *alpha_y = posY;

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

void colourThresholding2(std::shared_ptr<cv::Mat> imageToProcess, bool *processingDone, int *alpha_x, int *alpha_y, int mLowH, int mHighH, int gLowH, int gHighH)
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

    // RED
    //int mLowH = 317/2;
    //int mHighH = 345/2;

    int mLowS = 50; 
    int mHighS = 255;

    int mLowV = 50;
    int mHighV = 255;

    // 317 - 345
    // 27 - 65
    // 62 - 100

    // RED
    //int gLowH = 72/2;
    //int gHighH = 97/2;

    int gLowS = 50; 
    int gHighS = 255;

    int gLowV = 50;
    int gHighV = 255;

    // 72 - 97
    // 38 - 83
    // 55 - 80

    // GREEN
    int fLowH = 80/2;
    int fHighH = 140/2;

    int fLowS = 50; 
    int fHighS = 255;

    int fLowV = 50;
    int fHighV = 255;

    //Create a black image with the size as the camera output
    Mat imgLines = Mat::zeros(imgOriginal.size(), CV_8UC3);;

    //Threshold the image
    Mat imgThresholded1;
    inRange(imgOriginal, Scalar(mLowH, mLowS, mLowV), Scalar(mHighH, mHighS, mHighV), imgThresholded1);
    Mat imgThresholded2;
    inRange(imgOriginal, Scalar(gLowH, gLowS, gLowV), Scalar(gHighH, gHighS, gHighV), imgThresholded2);  
    Mat imgThresholded3;
    inRange(imgOriginal, Scalar(fLowH, fLowS, fLowV), Scalar(fHighH, fHighS, fHighV), imgThresholded3); 

    Mat imgThresholded;
    imgThresholded = imgThresholded1 + imgThresholded2 + imgThresholded3; 

    //morphological opening (removes small objects from the foreground)
    erode(imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(15, 15)) );
    dilate(imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(15, 15)) ); 

    //morphological closing (removes small holes from the foreground)
    dilate(imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(15, 15)) ); 
    erode(imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(15, 15)) );

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

      *alpha_x = posX;
      *alpha_y = posY;

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



void colourThresholding2Save(std::shared_ptr<cv::Mat> imageToProcess, bool *processingDone, int *alpha_x, int *alpha_y, int mLowH, int mHighH, int gLowH, int gHighH, int counter)
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

    // RED
    //int mLowH = 317/2;
    //int mHighH = 345/2;

    int mLowS = 50; 
    int mHighS = 255;

    int mLowV = 50;
    int mHighV = 255;

    // 317 - 345
    // 27 - 65
    // 62 - 100

    // RED
    //int gLowH = 72/2;
    //int gHighH = 97/2;

    int gLowS = 70; 
    int gHighS = 255;

    int gLowV = 70;
    int gHighV = 255;

    // 72 - 97
    // 38 - 83
    // 55 - 80

    int yLowH = 52;
    int yHighH = 60;

    int yLowS = 50; 
    int yHighS = 255;

    int yLowV = 50;
    int yHighV = 255;

    //Create a black image with the size as the camera output
    Mat imgLines = Mat::zeros(imgOriginal.size(), CV_8UC3);;

    //Threshold the image
    Mat imgThresholded1;
    inRange(imgOriginal, Scalar(mLowH, mLowS, mLowV), Scalar(mHighH, mHighS, mHighV), imgThresholded1);
    Mat imgThresholded2;
    inRange(imgOriginal, Scalar(gLowH, gLowS, gLowV), Scalar(gHighH, gHighS, gHighV), imgThresholded2);
    Mat imgThresholded3;
    inRange(imgOriginal, Scalar(yLowH, yLowS, yLowV), Scalar(yHighH, yHighS, yHighV), imgThresholded3);   

    Mat imgThresholded;
    imgThresholded = imgThresholded1 + imgThresholded2 + imgThresholded3; 

    //morphological opening (removes small objects from the foreground)
    erode(imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(14, 14)) );
    dilate(imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(14, 14)) ); 

    //morphological closing (removes small holes from the foreground)
    dilate(imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(14, 14)) ); 
    erode(imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(14, 14)) );

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

      *alpha_x = posX;
      *alpha_y = posY;

      string position_s = "Position: " + to_string(posX) + ", " + to_string(posY);
      putText(imgThresholded, position_s, Point(0, 125), FONT_HERSHEY_PLAIN, 3.0, Scalar(255, 255, 255), 2, 8, false);
      circle(imgThresholded, Point(posX, posY), 30, Scalar(0,0,255), 4, 8, 0);

    }

    putText(imgThresholded, time_s, Point(0, 25), FONT_HERSHEY_PLAIN, 3.0, Scalar(255, 255, 255), 2, 8, false);

    
    if (counter % 4 == 0 && dArea > 1000000) {
        imwrite("./good/thres - " + time_s  + ".jpg", imgThresholded);
	imwrite("./good/color - " + time_s  + ".jpg", (*imageToProcess));
	cout << "AREA: ";
	cout << dArea << endl;
    }
    ////

    // Output Image
    imgThresholded.copyTo(*imageToProcess);

    *processingDone = true;
}




void contrast(shared_ptr<Mat> imageToProcess, bool *processingDone)
{
    //Capture a temporary image from the camera
    Mat imgCont;
    (*imageToProcess).convertTo(imgCont, CV_32S);

    double beta = 200.0; // CANNOT GO OVER 255
    int kappa = 259; // stay at 259
    double contrast_factor = (kappa * (beta + 255)) / (255 *(kappa - beta));
    
    Scalar scale(128, 128, 128);

    imgCont -= scale;
    imgCont *= contrast_factor;
    imgCont += scale;

    imgCont.convertTo(imgCont, CV_8U);

    // Output Image
    imgCont.copyTo(*imageToProcess);

    *processingDone = true;
}







void grayscale(shared_ptr<Mat> imageToProcess, bool *processingDone)
{
    Mat gray;
    cvtColor(*imageToProcess, gray, COLOR_BGR2GRAY); // Convert to grayscale

    // Output Image
    gray.copyTo(*imageToProcess);

    *processingDone = true;
}





void saveIm(shared_ptr<Mat> imageToProcess, bool *processingDone, int counter)
{
    // Time
    time_t tt = time(NULL);
    string time_s = ctime(&tt);
    time_s = "Time: " + time_s.substr(0, time_s.size()-1);

    if (counter % 500 == 0) imwrite("./a/AHMED - " + time_s + ".jpg", (*imageToProcess));

    *processingDone = true;
}
