#include <opencv2/opencv.hpp>

#include <vector>
#include <numeric>
using namespace cv;
using namespace std;

template<class T>
class CompareMin
{
public:
	T operator() (vector<T> data, vector<double> ele) 
	{
		return *std::min_element(data.begin(), data.end());
	}
	vector<T> data[3];
};

template<class T>
class CompareMax
{
public:
	T operator() (vector<T> data, vector<double> ele) 
	{
		return *std::max_element(data.begin(), data.end());
	}
	vector<T> data[3];
};

template<class T>
class CompareMean
{
public:
	T operator() (vector<T> data, vector<double> ele) 
	{
		T acc = 0;
		for (int i = 0; i < data.size(); ++i) {
			acc += data[i]*(1.0);
		}
		return acc / data.size();
	}
	vector<T> data[3];
};

template<class T>
class CompareMedian
{
public:
	T operator() (vector<T> data, vector<double> ele) 
	{
		sort(data.begin(), data.end());
		int median = (data.size()>>1);
		return data[median];
	}
	vector<T> data[3];
};

template<class T>
class CompareGaussian
{
public:
	T operator() (vector<T> data, vector<double> ele) 
	{
		T acc = 0;
		for (int i = 0; i < data.size(); ++i) {
			acc += data[i]*ele[i];
		}
		return acc;
	}
	vector<T> data[3];
};

Mat mgetGaussianKernel(const int size, const double sigma);
template<class Compare>
void convolution(Mat& src, Mat& dst, Mat& ele, Compare comp);

void merode(Mat& src, Mat& dst, Mat& ele);
void mdilate(Mat& src, Mat& dst, Mat& ele);
void mopen(Mat& src, Mat& dst, Mat& ele);
void mclose(Mat& src, Mat& dst, Mat& ele);
void mtopHat(Mat& src, Mat& dst, Mat& ele);
void mblackHat(Mat& src, Mat& dst, Mat& ele);
void mgradient(Mat& src, Mat& dst, Mat& ele);
void mblur(Mat& src, Mat& dst, Size size);
void mmedianBlur(Mat& src, Mat& dst, int ksize);
void mGaussianBlur(Mat& src, Mat& dst, Size ksize);

void mfloodFill(Mat& image, Point seedPoint, Scalar newVal,
            Scalar loDiff = Scalar(), Scalar upDiff = Scalar(), int flags = 4 );

void mpyrUp(Mat& src, Mat& dst, Size ksize);
void mpyrDown(Mat& src, Mat& dst, Size ksize);

void mresizeNearest(Mat& src, Mat& dst, double scale_x, double scale_y);
void mresizeLinear(Mat& src, Mat& dst, double scale_x, double scale_y);
void mresizeArea(Mat& src, Mat& dst, double scale_x, double scale_y);
void mresize(Mat& src, Mat& dst, Size ksize, double fx = 0, double fy = 0, int interpolation = INTER_LINEAR);

double thresholdBinary(double srcval, double thresh, int retval);
double thresholdBinaryInv(double srcval, double thresh, int retval);
double thresholdThunc(double srcval, double thresh, int retval);
double thresholdTozero(double srcval, double thresh, int retval);
double thresholdTozeroInv(double srcval, double thresh, int retval);
void mthreshold(Mat& src, Mat& dst, double thresh, double maxval, int type);

double gauss(float x, float sigma);
double dgauss(float x, float sigma);
bool nonmax_suppress(double theta, Mat &g_mat, Point anchor);
void mCanny(Mat& src, Mat& dst, double threshold1, double threshold2, int apertureSize = 2);

void mHoughLines(Mat& src, vector<Vec2f>& dst, double rho, double theta, int threshold, double srn=0, double stn=0);