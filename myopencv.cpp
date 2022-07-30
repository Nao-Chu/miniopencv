#include "myopencv.h"

#include <iostream>
using namespace std;

Mat mGetGaussianKernel(const int size, const double sigma)
{
    double **gaus=new double *[size];
    for(int i=0;i<size;i++)
    {
        gaus[i]=new double[size];  //动态生成矩阵
    }
    Mat Kernel(size,size,CV_64FC1,Scalar(0));
    const double PI=4.0*atan(1.0); //圆周率π赋值
    int center=size/2;
    double sum=0;
    for(int i=0;i<size;i++)
    {
        for(int j=0;j<size;j++)
        {
            gaus[i][j]=(1/(2*PI*sigma*sigma))*exp(-((i-center)*(i-center)+(j-center)*(j-center))/(2*sigma*sigma));//二维高斯函数
            sum+=gaus[i][j];
        }
    }
 
 
    for(int i=0;i<size;i++)
    {
        for(int j=0;j<size;j++)
        {
            gaus[i][j]/=sum;
            Kernel.at<double>(i,j) = gaus[i][j];//将数组转换为Mat
 
        }
    }
    return Kernel;
}

template<class Compare>
void convolution(Mat& src, Mat& dst, Mat& ele, Compare comp)
{
	dst = Mat::zeros(src.size(), src.type());

	int cx = ele.rows >> 1; 	
	int cy = ele.cols >> 1;	
	for (int i = 0; i < src.rows; ++i) 
	{
		for (int j = 0; j < src.cols; ++j) 
		{
			comp.data[0].clear();
			comp.data[1].clear();
			comp.data[2].clear();
			vector<double> ele_v;
			for (int k = -cx; k <= cx; ++k) 
			{
				for (int l = -cy; l <= cy; ++l) 
				{

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
					ele_v.push_back(ele.at<double>(k+cx,l+cy));

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

void mopen(Mat& src, Mat& dst, Mat& ele)
{
	Mat temp;
	merode(src, temp, ele);
	mdilate(temp, dst, ele);
}

void mclose(Mat& src, Mat& dst, Mat& ele)
{
	Mat temp;
	mdilate(src, temp, ele);
	merode(temp, dst, ele);
}

void mtopHat(Mat& src, Mat& dst, Mat& ele)
{
	mopen(src, dst, ele);
	dst = src - dst;
}

void mblackHat(Mat& src, Mat& dst, Mat& ele)
{
	mclose(src, dst, ele);
	dst = dst - src;
}

void mgradient(Mat& src, Mat& dst, Mat& ele)
{
	Mat temp;
	mdilate(src, temp, ele);
	merode(src, dst, ele);
	dst = temp - dst;
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
	double sigma = 0.3*((ksize.height-1)*0.5-1)+0.8;
	Mat ele = mGetGaussianKernel(ksize.height, sigma);
	CompareGaussian<double> func;
	convolution(src, dst, ele, func);
}

bool isflood(Scalar diff, Scalar loDiff, Scalar upDiff)
{
	for (int i = 0; i < 3; ++i) {
		if(!(diff[i] >= loDiff[i] && diff[i] <= upDiff[i] )) {
			return false;
		}
	}
	return true;
}

void mfloodFill(Mat& image, Point seedPoint, Scalar newVal, Scalar loDiff, Scalar upDiff, int flags)
{
	queue<int> root;
	int n = image.rows;
	int m = image.cols;
	int code = seedPoint.x * m + seedPoint.y;
	root.push(code);
	
	vector<int> dx, dy;
	if (flags == 4) {
		dx = {-1, 1, 0, 0};
		dy = {0, 0, -1, 1};
	} else {
		dx = {-1, 1, 0, 0, -1, -1, 1, 1};
		dy = {0, 0, -1, 1, -1, 1, -1, 1};
	}

	vector<vector<int>> fill(n, vector<int>(m,0));
	Scalar mloDiff, mupDiff;
	while (!root.empty()) {
		code = root.front();
		root.pop();
		int i = code / m;
		int j = code % m;
		for (int k = 0; k < 3; ++k) {
			mloDiff[k] = image.at<Vec3b>(i, j)[k] - loDiff[k];
			mupDiff[k] = image.at<Vec3b>(i, j)[k] + upDiff[k];
		}	

		for (int k = 0; k < dx.size(); ++k) {
			int mx = dx[k] + i;
			int my = dy[k] + j;
			if (mx < 0 || mx >= n || my < 0 || my >= m || fill[mx][my] == 1) {
				continue;
			}

			Scalar diff;
			diff[0] = image.at<Vec3b>(mx, my)[0];
			diff[1] = image.at<Vec3b>(mx, my)[1];
			diff[2] = image.at<Vec3b>(mx, my)[2];
			if (isflood(diff, mloDiff, mupDiff)) {	
				fill[mx][my] = 1;
			 	root.push(mx * m + my);
			}
		}
	}

	for (int i = 0; i < fill.size(); ++i)
	{
		for (int j = 0; j < fill[0].size(); ++j) 
		{
			if (fill[i][j] == 0) {
				continue;
			}
			image.at<Vec3b>(i, j) = Vec3b(newVal[0], newVal[1], newVal[2]);
		}
		
	}
}