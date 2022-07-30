#include <stdio.h>
#include <iostream>
#include "myopencv.h"
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;
int main() 
{
    //Mat srcImage = imread("test.png");
    Mat srcImage = imread("image/erode.png");
	//Mat test = Mat(300, 300, CV_8UC1); 
	//for (int i = 100; i < 200; ++i) {
	//for (int j = 100; j < 200; ++j) {
	//	test.at<uchar>(i,j) = 255;
	//}
	//}
    //Mat srcImage = test;
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
    // morphologyEx(srcImage, dstImage, MORPH_GRADIENT, element);
    // mgradient(srcImage, mdstImage, element);
    Rect ccomp;
    dstImage = srcImage.clone();
    mdstImage = srcImage.clone();
    floodFill(dstImage, Point(100,100), Scalar(255,0,0), &ccomp, Scalar(10,10,10), Scalar(10,10,10));
    mfloodFill(mdstImage, Point(100,100), Scalar(255,0,0), Scalar(10,10,10), Scalar(10,10,10));
    
//    imwrite("1erode.png", test);
	cout << "dst: " << dstImage.channels() << endl;
	cout << "mdst: " << mdstImage.channels() << endl;
	cout << "ele: " << element.channels() << endl;
    imshow("src", srcImage);
    imshow("dst", dstImage);
    imshow("mdst", mdstImage);
    //imshow("ele", element);

	Mat c;
	compare(dstImage, mdstImage, c, CMP_EQ);
    imshow("c", c);
	waitKey(0);

    return 0;
}
