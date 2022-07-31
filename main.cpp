#include <stdio.h>
#include <iostream>
#include "myopencv.h"
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;
int main() 
{
    // Mat srcImage = imread("image/erode.png");
    // resize(srcImage, srcImage, Size(320,480));
    // Mat srcImage = imread("image/image.jpg");
	Mat test = Mat(1, 1, CV_8UC1); 
	for (int i = 0; i < 1; ++i) {
	for (int j = 0; j < 1; ++j) {
		test.at<uchar>(i,j) = j+1;
	}
	}
    Mat srcImage = test;
    imshow("src", srcImage);
    Mat element = getStructuringElement(MORPH_RECT, Size(3, 3));
 
    Mat dstImage;
    Mat mdstImage;
    //erode(srcImage, dstImage, element);
    //merode(srcImage, mdstImage, element);
    //dilate(srcImage, dstImage, element);
    //mdilate(srcImage, mdstImage, element);
    //blur(srcImage, dstImage, Size(15,15));
    //mblur(srcImage, mdstImage, Size(15,15));
    // GaussianBlur(srcImage, dstImage, Size(5,5), 0, 0);
    // mGaussianBlur(srcImage, mdstImage, Size(5,5));
    // morphologyEx(srcImage, dstImage, MORPH_GRADIENT, element);
    // mgradient(srcImage, mdstImage, element);
    // Rect ccomp;
    // dstImage = srcImage.clone();
    // mdstImage = srcImage.clone();
    // floodFill(dstImage, Point(100,100), Scalar(255,0,0), &ccomp, Scalar(10,10,10), Scalar(10,10,10));
    // mfloodFill(mdstImage, Point(100,100), Scalar(255,0,0), Scalar(10,10,10), Scalar(10,10,10));
    // pyrUp(srcImage, dstImage, Size(srcImage.cols*2, srcImage.rows*2));
    // mpyrUp(srcImage, mdstImage, Size(srcImage.cols*2, srcImage.rows*2));
    // expendSampleFunc(srcImage, mdstImage);
    // pyrDown(srcImage, dstImage, Size(srcImage.cols>>1, srcImage.rows>>1));
    // mpyrDown(srcImage, mdstImage, Size(srcImage.cols>>1, srcImage.rows>>1));
    resize(srcImage, dstImage, Size(), 2, 2, INTER_LINEAR);
    mresize(srcImage, mdstImage, Size(), 2, 2);

//    imwrite("1erode.png", test);
	cout << "dst: " << dstImage.channels() << endl;
	cout << "mdst: " << mdstImage.channels() << endl;
	cout << "ele: " << element.channels() << endl;
    imshow("src", srcImage);
    imshow("dst", dstImage);
    imshow("mdst", mdstImage);
    //imshow("ele", element);

    for (int i = 0; i < dstImage.rows; ++i) {
        for (int j = 0; j < dstImage.cols; ++j) {
            cout << (int)dstImage.at<uchar>(i,j) << " ";
        }
        cout << endl;
	}
    cout << endl;
    for (int i = 0; i < dstImage.rows; ++i) {
        for (int j = 0; j < dstImage.cols; ++j) {
            cout << (int)mdstImage.at<uchar>(i,j) << " ";
        }
        cout << endl;
	}
	Mat c;
	compare(dstImage, mdstImage, c, CMP_EQ);
    imshow("c", c);
	waitKey(0);

    return 0;
}
