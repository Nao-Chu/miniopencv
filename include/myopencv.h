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

template<class Compare>
void convolution(Mat& src, Mat& dst, Mat& ele, Compare comp);

void merode(Mat& src, Mat& dst, Mat& ele);
void mdilate(Mat& src, Mat& dst, Mat& ele);
void mblur(Mat& src, Mat& dst, Size size);
