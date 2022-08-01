#include "myopencv.h"

#include <iostream>

Mat mgetGaussianKernel(const int size, const double sigma)
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
	Mat ele = mgetGaussianKernel(ksize.height, sigma);
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

void mpyrUp(Mat& src, Mat& dst, Size ksize)
{
	Mat temp = Mat::zeros(ksize, src.type());
	int x_population = ksize.height/src.rows;
	int y_population = ksize.width/src.cols;
	for (int i = 0; i < src.rows; ++i) 
	{
		for (int j = 0; j < src.cols; ++j) 
		{
			if (temp.channels() == 1) {
				temp.at<uchar>(i*x_population,j*y_population) = src.at<uchar>(i,j);
				
			} else {
				temp.at<Vec3b>(i*x_population,j*y_population) = src.at<Vec3b>(i,j);
			}
		}
	}
	
	// double sigma = 0.3*((5-1)*0.5-1)+0.8;
	// Mat ele = mgetGaussianKernel(5, 100);

	Mat ele(Size(5, 5), CV_64FC1);
	float tempArr[5][5]={1,4,6,4,1,4,16,24,16,4,6,24,36,24,6,4,16,24,16,4,1,4,6,4,1};
	for (int i = 0;i<5;i++)
	{
		for (int j = 0; j<5; j++)
		{
			ele.at<double>(i,j) = tempArr[i][j] / 256;
		}
	}
	ele *= x_population*x_population;
	CompareGaussian<double> func;
	convolution(temp, dst, ele, func);
}

void mpyrDown(Mat& src, Mat& dst, Size ksize)
{
	Mat temp;
	mGaussianBlur(src, temp, Size(5,5));
	int x_population = src.rows/ksize.height;
	int y_population = src.cols/ksize.width;
	dst = Mat::zeros(ksize, src.type());
	for (int i = 0; i < dst.rows; ++i) 
	{
		for (int j = 0; j < dst.cols; ++j) 
		{
			if (dst.channels() == 1) {
				dst.at<uchar>(i,j) = temp.at<uchar>(i*x_population,j*y_population);
				
			} else {
				dst.at<Vec3b>(i,j) = temp.at<Vec3b>(i*x_population,j*y_population);
			}
		}
	}
}

void mresizeNearest(Mat& src, Mat& dst, double scale_x, double scale_y)
{
	for (int j = 0; j < dst.rows; ++j)
	{
		int sy = cvFloor(j * scale_y);
		sy = std::min(sy, src.rows - 1);
		for (int i = 0; i < dst.cols; ++i)
		{
			int sx = cvFloor(i * scale_x);
			sx = std::min(sx, src.cols - 1);

			if (dst.channels() == 1) {
				dst.at<uchar>(j, i) = src.at<uchar>(sy, sx);
				
			} else {
				dst.at<Vec3b>(j, i) = src.at<Vec3b>(sy, sx);
			}
		}
	}

}

void mresizeLinear(Mat& src, Mat& dst, double scale_x, double scale_y)
{
	int dx[] = {0, 1, 0, 1};
	int dy[] = {0, 0, 1, 1};

	uchar* dataDst = dst.data;
	int stepDst = dst.step;
	uchar* dataSrc = src.data;
	int stepSrc = src.step;

	for (int j = 0; j < dst.rows; ++j)
	{
		float fy = (float)((j + 0.5) * scale_y - 0.5);
		int sy = cvFloor(fy);
		fy -= sy;
		sy = std::min(sy, src.rows - 1);
		sy = std::max(0, sy);

		short cbufy[2];
		cbufy[0] = cv::saturate_cast<short>((1.f - fy) * 2048);
		cbufy[1] = 2048 - cbufy[0];

		for (int i = 0; i < dst.cols; ++i)
		{
			float fx = (float)((i + 0.5) * scale_x - 0.5);
			int sx = cvFloor(fx);
			fx -= sx;
 
			if (sx < 0) {
				fx = 0, sx = 0;
			}
			if (sx >= src.cols - 1) {
				fx = 0, sx = src.cols - 1;
			}
 
			short cbufx[2];
			cbufx[0] = cv::saturate_cast<short>((1.f - fx) * 2048);
			cbufx[1] = 2048 - cbufx[0];


			for (int k = 0; k < src.channels(); ++k)
			{
				*(dataDst+ j*stepDst + 3*i + k) = (*(dataSrc + sy*stepSrc + 3*sx + k) * cbufx[0] * cbufy[0] + 
					*(dataSrc + (sy+1)*stepSrc + 3*sx + k) * cbufx[0] * cbufy[1] + 
					*(dataSrc + sy*stepSrc + 3*(sx+1) + k) * cbufx[1] * cbufy[0] + 
					*(dataSrc + (sy+1)*stepSrc + 3*(sx+1) + k) * cbufx[1] * cbufy[1]) >> 22;
			}
		}
	}
}

void mresizeArea(Mat& src, Mat& dst, double scale_x, double scale_y)
{
	uchar* dataDst = dst.data;
	int stepDst = dst.step;
	uchar* dataSrc = src.data;
	int stepSrc = src.step;
	int fx = cvFloor(scale_x);
	int fy = cvFloor(scale_y);

	for (int j = 0; j < dst.rows; ++j)
	{
		for (int i = 0; i < dst.cols; ++i)
		{
			for (int k = 0; k < src.channels(); ++k)
			{
				double box = 0;
				for (int m = j*fy; m < ((j+1)*fy); ++m) {
					for (int n = i*fx; n < ((i+1)*fx); ++n) {
						box += *(dataSrc + m*stepSrc + 3*n + k);
						double temp = *(dataSrc + m*stepSrc + 3*n + k);
					}
				}
				box /= (fy*fx);
				*(dataDst+ j*stepDst + 3*i + k) = cvFloor(box);
			}
		}
	}
}

void mresize(Mat& src, Mat& dst, Size ksize, double fx, double fy, int interpolation)
{
	if (ksize.width == 0 || ksize.height == 0) {
		ksize = Size(fx*src.cols, fy*src.rows);
	} else {
		fx = ksize.width*1.0/src.cols;
		fy = ksize.height*1.0/src.rows;
	}

	if (fx == 1 && fy == 1) {
		dst = src;
		return;
	}

	int ifx = saturate_cast<int>(fx);
	int ify = saturate_cast<int>(fy);

	bool is_area_fast = std::abs(fx - ifx) < DBL_EPSILON && std::abs(fy - ify) < DBL_EPSILON;

	if (interpolation == INTER_LINEAR && is_area_fast && fx == 2 && fy == 2) {
		interpolation = INTER_AREA;
	}

	double scale_x = 1./fx;
	double scale_y = 1./fy;
	if (interpolation == INTER_AREA && (fx > 1 || fy > 1)) {
		interpolation = INTER_LINEAR;
	}


	dst = Mat::zeros(ksize, src.type());
	if (interpolation == INTER_NEAREST) {
		mresizeNearest(src, dst, scale_x, scale_y);
	} else if (interpolation == INTER_LINEAR) {
		mresizeLinear(src, dst, scale_x, scale_y);
	} else if (interpolation == INTER_AREA) {
		mresizeArea(src, dst, scale_x, scale_y);
	}
}