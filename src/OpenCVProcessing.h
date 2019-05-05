/*
 * OpenCVProcessing.h
 *
 *  Created on: Apr 15, 2019
 *      Author: slascos
 */
#include <memory>

#ifndef SRC_OPENCVPROCESSING_H_
#define SRC_OPENCVPROCESSING_H_

#include "opencv2/opencv.hpp"

void colourThresholding(std::shared_ptr<cv::Mat> imageToProcess, bool *processingDone, int *alpha_x, int *alpha_y);
void colourThresholding2(std::shared_ptr<cv::Mat> imageToProcess, bool *processingDone, int *alpha_x, int *alpha_y, int mLowH, int mHighH, int gLowH, int gHighH);
void contrast(std::shared_ptr<cv::Mat> imageToProcessPtr, bool *processingDone);
void grayscale(std::shared_ptr<cv::Mat> imageToProcessPtr, bool *processingDone);

#endif /* SRC_OPENCVPROCESSING_H_ */
