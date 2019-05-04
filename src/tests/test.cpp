#include <iostream>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace cv;
using namespace std;

int main( int argc, char** argv )
{
  VideoCapture cap(1); //capture the video from webcam

  if ( !cap.isOpened() )  // if not success, exit program
  {
    cout << "Cannot open the web cam" << endl;
    return -1;
  }

  namedWindow("Control", CV_WINDOW_AUTOSIZE); //create a window called "Control"

  // Magenta
  int mLowH = 310;
  int mHighH = 350;

  int mLowS = 25; 
  int mHighS = 70;

  int mLowV = 60;
  int mHighV = 120;

  // 317 - 345
  // 27 - 65
  // 62 - 100

  // Green
  int gLowH = 70;
  int gHighH = 100;

  int gLowS = 35; 
  int gHighS = 85;

  int gLowV = 50;
  int gHighV = 100;

  // 72 - 97
  // 38 - 83
  // 55 - 80

  //Create trackbars in "Control" window
  createTrackbar("LowH", "Control", &mLowH, 179); //Hue (0 - 179)
  createTrackbar("HighH", "Control", &mHighH, 179);

  createTrackbar("LowS", "Control", &mLowS, 255); //Saturation (0 - 255)
  createTrackbar("HighS", "Control", &mHighS, 255);

  createTrackbar("LowV", "Control", &mLowV, 255);//Value (0 - 255)
  createTrackbar("HighV", "Control", &mHighV, 255);

  int iLastX = -1; 
  int iLastY = -1;

  //Capture a temporary image from the camera
  Mat imgTmp;
  cap.read(imgTmp); 

  //Create a black image with the size as the camera output
  Mat imgLines = Mat::zeros( imgTmp.size(), CV_8UC3 );;


  while (true)
  {
    Mat imgOriginal;

    bool bSuccess = cap.read(imgOriginal); // read a new frame from video

    if (!bSuccess) //if not success, break loop
    {
      cout << "Cannot read a frame from video stream" << endl;
      break;
    }

    // Convert to HSV
    Mat imgHSV;
    cvtColor(imgOriginal, imgHSV, COLOR_BGR2HSV); //Convert the captured frame from BGR to HSV

    //Threshold the image
    Mat imgThresholded1;
    inRange(imgHSV, Scalar(mLowH, mLowS, mLowV), Scalar(mHighH, mHighS, mHighV), imgThresholded1);
    Mat imgThresholded2;
    inRange(imgHSV, Scalar(gLowH, gLowS, gLowV), Scalar(gHighH, gHighS, gHighV), imgThresholded2);  

    Mat imgThresholded;
    imgThresholded = imgThresholded1 + imgThresholded2;

    //morphological opening (removes small objects from the foreground)
    erode(imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)) );
    dilate( imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)) ); 

    //morphological closing (removes small holes from the foreground)
    dilate( imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)) ); 
    erode(imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)) );

    //Calculate the moments of the thresholded image
    Moments oMoments = moments(imgThresholded);

    double dM01 = oMoments.m01;
    double dM10 = oMoments.m10;
    double dArea = oMoments.m00;

    // if the area <= 10000, I consider that the there are no object in the image and it's because of the noise, the area is not zero 
    if (dArea > 10000)
    {
      //calculate the position of the ball
      int posX = dM10 / dArea;
      int posY = dM01 / dArea;        

      circle(imgLines, Point(posX, posY), 5, Scalar(0,0,255), 1, 8, 0);

      // if (iLastX >= 0 && iLastY >= 0 && posX >= 0 && posY >= 0)
      // {
      //   //Draw a red line from the previous point to the current point
      //   line(imgLines, Point(posX, posY), Point(iLastX, iLastY), Scalar(0,0,255), 2);
      // } 

      iLastX = posX;
      iLastY = posY;
    }

    imshow("Thresholded Image", imgThresholded); //show the thresholded image

    imgOriginal = imgOriginal + imgLines;
    imshow("Original", imgOriginal); //show the original image

    if (waitKey(30) == 27) //wait for 'esc' key press for 30ms. If 'esc' key is pressed, break loop
    {
      cout << "esc key is pressed by user" << endl;
      break; 
    }

  }

  return 0;

}
