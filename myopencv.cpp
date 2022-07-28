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
			uchar convolution_v[3] = {comp.initv, comp.initv, comp.initv};
			for (int k = -cx; k <= cx; ++k) {
				for (int l = -cy; l <= cy; ++l) {

					int dx = i + k;	
					int dy = j + l;	
					if (dx < 0 || dx >= src.rows|| 
						dy < 0 || dy >= src.cols) {
						
						continue;
					}

					if (dst.channels() == 1) {
						src.at<uchar>(dx,dy) = comp(convolution_v[0], src.at<uchar>(dx,dy));
					} else {
						convolution_v[0] = comp(convolution_v[0], src.at<Vec3b>(dx,dy)[0]);
						convolution_v[1] = comp(convolution_v[1], src.at<Vec3b>(dx,dy)[1]);
						convolution_v[2] = comp(convolution_v[2], src.at<Vec3b>(dx,dy)[2]);
					}

				}
				
			}
			
			if (dst.channels() == 1) {
				dst.at<uchar>(i,j) = convolution_v[0];
			} else {
				dst.at<Vec3b>(i,j)[0] = convolution_v[0];
				dst.at<Vec3b>(i,j)[1] = convolution_v[1];
				dst.at<Vec3b>(i,j)[2] = convolution_v[2];
			}

		}
	}
}
void merode(Mat& src, Mat& dst, Mat& ele)
{
	CompareMin<uchar> func(255);
	convolution(src, dst, ele, func);
}

void mdilate(Mat& src, Mat& dst, Mat& ele)
{
	CompareMax<uchar> func(0);
	convolution(src, dst, ele, func);
}