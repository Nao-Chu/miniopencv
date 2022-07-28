#include "myopencv.h"

#include <iostream>
using namespace std;

void merode(Mat& src, Mat& dst, Mat& ele)
{
	dst = Mat::zeros(src.size(), src.type());
	int cx = ele.rows >> 1;	
	int cy = ele.cols >> 1;	
	for (int i = 0; i < src.rows; ++i) {
		for (int j = 0; j < src.cols; ++j) {
			bool is_full = true;
			uchar min_v[3] = {255, 255, 255};
			for (int k = -cx; k <= cx; ++k) {
				for (int l = -cy; l <= cy; ++l) {

					int dx = i + k;	
					int dy = j + l;	
					if (dx < 0 || dx >= src.rows|| 
						dy < 0 || dy >= src.cols) {
						
						continue;
					}

					int v = ele.at<uchar>(k+cx, l+cy); 
					if (dst.channels() == 1) {
						is_full = src.at<uchar>(dx,dy) & v;
					} else {
						min_v[0] = std::min(min_v[0], src.at<Vec3b>(dx,dy)[0]);
						min_v[1] = std::min(min_v[1], src.at<Vec3b>(dx,dy)[1]);
						min_v[2] = std::min(min_v[2], src.at<Vec3b>(dx,dy)[2]);
					}

					if (!is_full) {
						break;
					}
				}
				if (!is_full) {
					break;
				}
				
			}
			
			if (is_full) {
				if (dst.channels() == 1) {
					dst.at<uchar>(i,j) = 255;
				} else {
					dst.at<Vec3b>(i,j)[0] = min_v[0];
					dst.at<Vec3b>(i,j)[1] = min_v[1];
					dst.at<Vec3b>(i,j)[2] = min_v[2];
				}
			}

		}
	}
}



void mdilate(Mat& src, Mat& dst, Mat& ele)
{
	dst = Mat::zeros(src.size(), src.type());
	int cx = ele.rows >> 1;	
	int cy = ele.cols >> 1;	
	for (int i = 0; i < src.rows; ++i) {
		for (int j = 0; j < src.cols; ++j) {
			bool is_full = false;
			uchar max_v[3] = {0, 0, 0};
			for (int k = -cx; k <= cx; ++k) {
				for (int l = -cy; l <= cy; ++l) {

					int dx = i + k;	
					int dy = j + l;	
					if (dx < 0 || dx >= src.rows|| 
						dy < 0 || dy >= src.cols) {
						
						continue;
					}

					int v = ele.at<uchar>(k+cx, l+cy); 
					if (dst.channels() == 1) {
						is_full = src.at<uchar>(dx,dy) & v;
					} else {
						max_v[0] = std::max(max_v[0], src.at<Vec3b>(dx,dy)[0]);
						max_v[1] = std::max(max_v[1], src.at<Vec3b>(dx,dy)[1]);
						max_v[2] = std::max(max_v[2], src.at<Vec3b>(dx,dy)[2]);
					}

					if (is_full) {
						break;
					}
				}
				if (is_full) {
					break;
				}
				
			}
			
			if (is_full && dst.channels() == 1) {
				dst.at<uchar>(i,j) = 255;
			} else if (dst.channels() == 3) {
				dst.at<Vec3b>(i,j)[0] = max_v[0];
				dst.at<Vec3b>(i,j)[1] = max_v[1];
				dst.at<Vec3b>(i,j)[2] = max_v[2];
			}
		}
	} 
	//dst = src;
}
