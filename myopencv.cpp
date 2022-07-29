#include "myopencv.h"

#include <iostream>
using namespace std;


template<class Compare>
void convolution(Mat& src, Mat& dst, Mat& ele, Compare comp)
{
	dst = Mat::zeros(src.size(), src.type());

	int cx = ele.rows >> 1; 	
	int cy = ele.cols >> 1;	
	for (int i = 0; i < src.rows; ++i) {
		for (int j = 0; j < src.cols; ++j) {
			comp.data[0].clear();
			comp.data[1].clear();
			comp.data[2].clear();
			vector<double> ele_v;
			for (int k = -cx; k <= cx; ++k) {
				for (int l = -cy; l <= cy; ++l) {

					int dx = i + k;	
					int dy = j + l;	
					if (dx < 0 || dx >= src.rows|| 
						dy < 0 || dy >= src.cols) {
						
						continue;
					}

					if (dst.channels() == 1) {
						comp.data[0].push_back(src.at<uchar>(dx,dy));
						
					} else {
						comp.data[0].push_back(src.at<Vec3b>(dx,dy)[0]);
						comp.data[1].push_back(src.at<Vec3b>(dx,dy)[1]);
						comp.data[2].push_back(src.at<Vec3b>(dx,dy)[2]);
					}
					ele_v.push_back((double)ele.at<uchar>(k+cx,l+cy));

				}
				
			}
			
			if (dst.channels() == 1) {
				dst.at<uchar>(i,j) = comp(comp.data[0], ele_v);
			} else {
				dst.at<Vec3b>(i,j)[0] = comp(comp.data[0], ele_v);
				dst.at<Vec3b>(i,j)[1] = comp(comp.data[1], ele_v);
				dst.at<Vec3b>(i,j)[2] = comp(comp.data[2], ele_v);
			}

		}
	}
}

void merode(Mat& src, Mat& dst, Mat& ele)
{
	CompareMin<uchar> func;
	convolution(src, dst, ele, func);
}

void mdilate(Mat& src, Mat& dst, Mat& ele)
{
	CompareMax<uchar> func;
	convolution(src, dst, ele, func);
}

void mblur(Mat& src, Mat& dst, Size size)
{
	Mat ele = Mat::ones(size, CV_8UC1);
	CompareMean<double> func;
	convolution(src, dst, ele, func);
}

void mmedianBlur(Mat& src, Mat& dst, int ksize)
{
	Size size(ksize, ksize);
	Mat ele = Mat::ones(size, CV_8UC1);
	CompareMedian<double> func;
	convolution(src, dst, ele, func);
}

void mGaussianBlur(Mat& src, Mat& dst, Size ksize)
{
	Mat ele = Mat::ones(ksize, CV_8UC1);
	CompareGaussian<double> func;
	convolution(src, dst, ele, func);
}