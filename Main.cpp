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

void getMinimalPattern(const cv::Mat &input, unsigned int &sizeX, unsigned int &sizeY, const int minimalPatternSize = 10)
{
	const float epsilon = 0.00005;

	//make input gray scale 
	cv::Mat img;	
	cv::cvtColor(input, img, CV_RGB2GRAY);

	//calculate auto correlation
	cv::Mat ac;
	autocorrDFT(img, ac);
	cv::Mat_<float>& ptrAc = (cv::Mat_<float>&)ac;

	//search minimal pattern
	sizeX = 0;
	sizeY = 0;
	for (int x = minimalPatternSize; x < ac.size().width  - minimalPatternSize; x++)
	{
		if (ptrAc(0, x)/ptrAc(0,0) > ptrAc(0, sizeX)/ptrAc(0,0) + epsilon || sizeX == 0)
		{
			sizeX = x;	
		}
	}
	for (int y = minimalPatternSize; y < ac.size().height - minimalPatternSize; y++)
	{
		if (ptrAc(y, 0)/ptrAc(0,0) > ptrAc(sizeY, 0)/ptrAc(0,0) + epsilon || sizeY == 0)
		{
			sizeY = y;	
		}
	}
} 

int main (int argc, char** argv)
{

	cv::Mat src = cv::imread(argv[1]);

/*	for (int i = 0; i < src.size().width; i++)
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
*/

	unsigned int sizeX, sizeY;
	getMinimalPattern(src, sizeX, sizeY, atoi(argv[2]));	
	cv::Mat pattern = cv::Mat(src, cv::Rect(0, 0, sizeX, sizeY));
	cv::Mat repeatedPattern = cv::Mat(pattern.size().height * 3, pattern.size().width * 3, pattern.type());

	for (int y = 0; y < 3; y++)
	{
		for (int x = 0; x < 3; x++)
		{
			cv::Mat roi(repeatedPattern, cv::Rect(x * pattern.size().width, y * pattern.size().height, pattern.size().width, pattern.size().height));
			pattern.copyTo(roi);
		}
	
	}

	cv::startWindowThread();

	cv::namedWindow("Pattern", CV_WINDOW_AUTOSIZE);
	cv::imshow("Pattern", pattern);
	cv::waitKey();
	cv::namedWindow("RepeatedPattern", CV_WINDOW_AUTOSIZE);
	cv::imshow("RepeatedPattern", repeatedPattern);
	cv::waitKey();

	cv::destroyAllWindows();



	return 0;
}
