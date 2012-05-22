#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <iostream>

using namespace std;

/**

yx

00 01 02 03 ...
10 11 12 12 ...
20 21 22 21 ...
.  .  .  .
.  .  .  .
.  .  .  .
**/

double autocorr(int y, int x, cv::Mat &img)
{
	cv::Mat_<uchar>& ptrImg = (cv::Mat_<uchar>&)img;
	int N = img.size().width * img.size().height;
	
	int lag = y * img.size().width + x;

	double mean = cv::mean(img).val[0];

	double a = 0, b = 0;
	for (int i = 0; i < N; i++)
	{
		a += (ptrImg(i / img.size().width, i % img.size().width) - mean) * (ptrImg( ((i+lag) % N) / img.size().width, ((i+lag) % N) % img.size().width) - mean);
		b += (ptrImg(i / img.size().width, i % img.size().width) - mean) * (ptrImg(i / img.size().width, i % img.size().width) - mean);
	}

	
	return a/b;
}

void autocorrDFT(const cv::Mat &img, cv::Mat &dst)
{
	cv::Mat fImg;
	img.convertTo(fImg, CV_32FC1);
	
	cv::Size dftSize;
	dftSize.width = cv::getOptimalDFTSize(img.cols);
	dftSize.height = cv::getOptimalDFTSize(img.rows);
	
	dst = cv::Mat(dftSize, CV_32FC1, cv::Scalar::all(0));
	
	cv::dft(fImg, dst);
	cv::mulSpectrums(dst, dst, dst, cv::DFT_INVERSE, true);
	cv::dft(dst, dst, cv::DFT_INVERSE | cv::DFT_SCALE);
} 

int main (int argc, char** argv)
{

	cv::Mat src = cv::imread(argv[1], 0); // 0 == grayscale

	for (int i = 0; i < src.size().width; i++)
	{
		for (int j = 0; j < src.size().height; j++)
		{
			double ac = autocorr(j,i, src);
			if (ac > atof(argv[2]))
			{
				cout<<i<<" "<<j<<" "<<ac <<endl;
			}
		}
	}
	cout<<"====================================================="<<endl;
	cv::Mat dst;
	autocorrDFT(src, dst);
	cv::Mat_<float>& ptrDst = (cv::Mat_<float>&)dst;
	//cv::Mat_<uchar>& ptrDst = (cv::Mat_<uchar>&)dst;
	for (int i = 0; i < dst.size().width; i++)
	{
		for (int j = 0; j < dst.size().height; j++)
		{
			double ac = ptrDst(j, i) / ptrDst(0,0);
			if (ac > atof(argv[2]))
			{
				cout<<i<<" "<<j<<" "<<ac <<endl;
			}
		}
	}

	return 0;
}
