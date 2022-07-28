#include "myopencv.h"

#include <iostream>
using namespace std;


template<class Compare>
void convolution(Mat& src, Mat& dst, vector<vector<int>> ele, Compare comp)
{
	dst = Mat::zeros(src.size(), src.type());

	if (ele.size() == 0) {
		return;
	}
	int cx = ele.size() >> 1;	
	int cy = ele[0].size() >> 1;	
	for (int i = 0; i < src.rows; ++i) {
		for (int j = 0; j < src.cols; ++j) {
			vector<vector<uchar> > convolution_v(3);
			vector<int> ele_v(3);
			for (int k = -cx; k <= cx; ++k) {
				for (int l = -cy; l <= cy; ++l) {

					int dx = i + k;	
					int dy = j + l;	
					if (dx < 0 || dx >= src.rows|| 
						dy < 0 || dy >= src.cols) {
						
						continue;
					}

					if (dst.channels() == 1) {
						convolution_v[0].push_back(src.at<uchar>(dx,dy));
						ele_v.push_back(ele[k+cx, l+cy]);
					} else {
						convolution_v[0].push_back(src.at<Vec3b>(dx,dy)[0]);
						convolution_v[1].push_back(src.at<Vec3b>(dx,dy)[1]);
						convolution_v[2].push_back(src.at<Vec3b>(dx,dy)[2]);
						ele_v.push_back(ele[k+cx, l+cy]);
					}

				}
				
			}
			
			if (dst.channels() == 1) {
				dst.at<uchar>(i,j) = comp(convolution_v[0], ele_v);
			} else {
				dst.at<Vec3b>(i,j)[0] = comp(convolution_v[0], ele_v);
				dst.at<Vec3b>(i,j)[1] = comp(convolution_v[1], ele_v);
				dst.at<Vec3b>(i,j)[2] = comp(convolution_v[2], ele_v);
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
	vector<vector<float>> ele;
	int ss = size.height*size.width;
	for (int i = 0; i < size.height; ++i) {
		for (int i = 0; i < size.width; ++i) {
			ele[i][j] = 1/ss;
		}
	}
	CompareMean<uchar> func;
	convolution(src, dst, ele, func);
}