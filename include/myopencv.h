#include <opencv2/opencv.hpp>

using namespace cv;


template<class T>
class CompareMin
{
public:
	CompareMin (T v)
	{
		initv = v;
	} 
	T operator() (T a, T b) {
		return (a < b) ? a : b;
	}
	T initv;
};

template<class T>
class CompareMax
{
public:
	CompareMax (T v)
	{
		initv = v;
	} 
	T operator() (T a, T b) {
		return (a > b) ? a : b;
	}
	T initv;
};

template<class Compare>
void convolution(Mat& src, Mat& dst, Mat& ele, Compare comp);

void merode(Mat& src, Mat& dst, Mat& ele);
void mdilate(Mat& src, Mat& dst, Mat& ele);
