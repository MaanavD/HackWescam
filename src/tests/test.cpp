#include <iostream>
#include <ctime>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace cv;
using namespace std;

// COLOUR TRACKER
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
  int mLowH = 40/2;
  int mHighH = 60/2;

  int mLowS = 70; 
  int mHighS = 255;

  int mLowV = 70;
  int mHighV = 255;

  // 317 - 345
  // 27 - 65
  // 62 - 100

  // Green
  int gLowH = 40/2;
  int gHighH = 60/2;

  int gLowS = 70; 
  int gHighS = 255;

  int gLowV = 70;
  int gHighV = 255;

  // 72 - 97
  // 38 - 83
  // 55 - 80

  //Create trackbars in "Control" window
  //createTrackbar("LowH", "Control", &mLowH, 179); //Hue (0 - 179)
  //createTrackbar("HighH", "Control", &mHighH, 179);

  //createTrackbar("LowS", "Control", &mLowS, 255); //Saturation (0 - 255)
  //createTrackbar("HighS", "Control", &mHighS, 255);

  //createTrackbar("LowV", "Control", &mLowV, 255);//Value (0 - 255)
  //createTrackbar("HighV", "Control", &mHighV, 255);

  int iLastX = -1; 
  int iLastY = -1;

  //Capture a temporary image from the camera
  Mat imgTmp;
  cap.read(imgTmp);
  //imgTmp = imread("DroneAreaA.jpg", IMREAD_COLOR);

  //Create a black image with the size as the camera output
  Mat imgLines = Mat::zeros( imgTmp.size(), CV_8UC3 );;

  int counter = 0;
  while (true)
  {
    Mat imgOriginal;

    // Time
    time_t tt = time(NULL);
    string time_s = ctime(&tt);
    time_s = "Time: " + time_s.substr(0, time_s.size()-1);

    bool bSuccess = cap.read(imgOriginal); // read a new frame from video
    //bool bSuccess = true;
    //imgOriginal = imread("DroneAreaA.jpg", IMREAD_COLOR);

    if (!bSuccess) //if not success, break loop
    {
      cout << "Cannot read a frame from video stream" << endl;
      break;
    }

    // Convert to HSV
    Mat imgHSV;
    cvtColor(imgOriginal, imgHSV, COLOR_BGR2HSV); //Convert the captured frame from BGR to HSV
    
    // Save normal image every x number of times
    if (counter % 10 == 0)
        imwrite("normal: " + time_s + ".jpg", imgOriginal);

    //Threshold the image
    Mat imgThresholded1;
    inRange(imgHSV, Scalar(mLowH, mLowS, mLowV), Scalar(mHighH, mHighS, mHighV), imgThresholded1);
    Mat imgThresholded2;
    inRange(imgHSV, Scalar(gLowH, gLowS, gLowV), Scalar(gHighH, gHighS, gHighV), imgThresholded2);  

    Mat imgThresholded;
    // imgThresholded = imgThresholded1 + imgThresholded2;
    //imgThresholded1.copyTo(imgThresholded);
    imgThresholded = imgThresholded1 + imgThresholded2; 

    //morphological opening (removes small objects from the foreground)
    erode(imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(11, 11)) );
    dilate( imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(11, 11)) ); 

    //morphological closing (removes small holes from the foreground)
    dilate( imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(11, 11)) ); 
    erode(imgThresholded, imgThresholded, getStructuringElement(MORPH_ELLIPSE, Size(11, 11)) );

    //Calculate the moments of the thresholded image
    Moments oMoments = moments(imgThresholded);

    double dM01 = oMoments.m01;
    double dM10 = oMoments.m10;
    double dArea = oMoments.m00;

    cout << dArea << "   " << dM01 << "    " << dM10 << endl;

    cvtColor(imgThresholded, imgThresholded, COLOR_GRAY2BGR);

    // if the area <= 10000, I consider that the there are no object in the image and it's because of the noise, the area is not zero 
    if (dArea > 100000)
    {
      //calculate the position of the ball
      int posX = dM10 / dArea;
      int posY = dM01 / dArea;        

      string position_s = "Position: " + to_string(posX) + ", " + to_string(posY);
      putText(imgThresholded, position_s, Point(0, 150), FONT_HERSHEY_PLAIN, 5.0, Scalar(255, 255, 255), 4, 8, false);

      //imgLines = Mat::zeros( imgTmp.size(), CV_8UC3 );;
      //circle(imgLines, Point(posX, posY), 30, Scalar(0,0,255), 10, 8, 0);
      circle(imgThresholded, Point(posX, posY), 30, Scalar(0,0,255), 10, 8, 0);

      iLastX = posX;
      iLastY = posY;
    }
    
    putText(imgThresholded, time_s, Point(0, 50), FONT_HERSHEY_PLAIN, 5.0, Scalar(255, 255, 255), 4, 8, false);
    imshow("Thresholded Image", imgThresholded); //show the thresholded image
    //imwrite("output_thres.jpg", imgThresholded);

    //imgOriginal = imgOriginal + imgLines;
    imshow("Original", imgOriginal); //show the original image
    //imwrite("output_target.jpg", imgOriginal);
    
    counter += 1;

    if (waitKey(30) == 27) //wait for 'esc' key press for 30ms. If 'esc' key is pressed, break loop
    {
      cout << "esc key is pressed by user" << endl;
      break; 
    }

  }

  return 0;

}
