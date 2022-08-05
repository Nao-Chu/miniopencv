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
    Mat srcImage = imread("image/image.jpg");
    resize(srcImage, srcImage, Size(320,320));
	// Mat test = Mat(1, 1, CV_8UC1); 
	// for (int i = 0; i < 1; ++i) {
	// for (int j = 0; j < 1; ++j) {
	// 	test.at<uchar>(i,j) = j+1;
	// }
	// }
    // Mat srcImage = test;
    imshow("src", srcImage);
    cvtColor(srcImage, srcImage, COLOR_BGR2GRAY);
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
    // resize(srcImage, dstImage, Size(), 2, 2, INTER_NEAREST);
    // mresize(srcImage, mdstImage, Size(), 2, 2, INTER_NEAREST);
    // Canny(srcImage, dstImage, 3, 9, 3);
    // mCanny(srcImage, mdstImage, 3, 9, 3);

    Mat midImage;
    Canny(srcImage, midImage, 50, 100, 3);
    cvtColor(midImage, dstImage, COLOR_GRAY2BGR);
    cvtColor(midImage, mdstImage, COLOR_GRAY2BGR);
    vector<Vec2f> lines;
    HoughLines(midImage, lines, 1, CV_PI/180, 100, 0, 0);
    vector<Vec2f> mlines;
    mHoughLines(srcImage, mlines, 1, CV_PI/180, 100, 0, 0);
    const int alpha = 1000;
    for (size_t i = 0; i < lines.size(); i++)
    {
        float rho = lines[i][0], theta = lines[i][1];
        double a = cos(theta), b = sin(theta);
        double x = rho * a, y = rho * b;
        Point pt1(cvRound(x + alpha * (-b)), cvRound(y + alpha * a));
        Point pt2(cvRound(x - alpha * (-b)), cvRound(y - alpha * a));
        line(dstImage, pt1, pt2, Scalar(255, 0, 255), 1, LINE_AA);
    }
    for (size_t i = 0; i < mlines.size(); i++)
    {
        float rho = mlines[i][0], theta = mlines[i][1];
        double a = cos(theta), b = sin(theta);
        double x = rho * a, y = rho * b;
        Point pt1(cvRound(x + alpha * (-b)), cvRound(y + alpha * a));
        Point pt2(cvRound(x - alpha * (-b)), cvRound(y - alpha * a));
        line(mdstImage, pt1, pt2, Scalar(255, 0, 255), 1, LINE_AA);
    }

//    imwrite("1erode.png", test);
    cout << "src: " << srcImage.channels() << endl;
	cout << "dst: " << dstImage.channels() << endl;
	cout << "mdst: " << mdstImage.channels() << endl;
	cout << "ele: " << element.channels() << endl;
    imshow("src", srcImage);
    imshow("dst", dstImage);
    imshow("mdst", mdstImage);
    //imshow("ele", element);

    // for (int i = 0; i < dstImage.rows; ++i) {
    //     for (int j = 0; j < dstImage.cols; ++j) {
    //         cout << (int)dstImage.at<uchar>(i,j) << " ";
    //     }
    //     cout << endl;
	// }
    // cout << endl;
    // for (int i = 0; i < dstImage.rows; ++i) {
    //     for (int j = 0; j < dstImage.cols; ++j) {
    //         cout << (int)mdstImage.at<uchar>(i,j) << " ";
    //     }
    //     cout << endl;
	// }
    if (dstImage.channels() == mdstImage.channels()) {
        Mat c;
        compare(dstImage, mdstImage, c, CMP_EQ);
        imshow("c", c);
    }
	waitKey(0);

    return 0;
}
