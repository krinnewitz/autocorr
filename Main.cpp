#include <stdlib.h>
#include "AutoCorr.hpp"
#include <stdio.h>
#include <math.h>
#include <iostream>

/**
 * \file	Main.cpp
 * \brief 	This is an implementation of the auto correlation 
 * 		function for images. The main goal is to verify how
 *		good this method is when trying to find patterns in
 *		textures.
 *
 * \author	Kim Oliver Rinnewitz, krinnewitz@uos.de
 */

using namespace std;


/**
 * \brief 	This auto correlation function should be used for personal
 *		understanding only. Do not try to use this function for
 *		productive jobs since it is terribly slow!
 *
 * \param	y 	The y shift to calculate the auto correlation for
 * \param	x 	The x shift to calculate the auto correlation for
 * \param	img	The image to use. Must be one channel gray scale. 
 * \return	The normed correlation value at the position determined by
 *		x and y
 */
double auto_corr(int y, int x, cv::Mat &img)
{
	cv::Mat_<uchar>& ptrImg = (cv::Mat_<uchar>&)img;
	int N = img.size().width * img.size().height;
	
	int lag = y * img.size().width + x;

	double mean = cv::mean(img).val[0];

	double a = 0, b = 0;
	for (int i = 0; i < N; i++)
	{
		a += 	  (ptrImg(i / img.size().width, i % img.size().width) - mean)
			* (ptrImg( ((i+lag) % N) / img.size().width, ((i+lag) % N) % img.size().width) - mean);
		b +=   	  (ptrImg(i / img.size().width, i % img.size().width) - mean)
			* (ptrImg(i / img.size().width, i % img.size().width) - mean);
	}

	
	return a/b;
}

/**
 * \brief 	
 *
 * \param 	img	The image to calculate the auto covariance for. Must be one channel
 *			gray scale.
 * \param	dst	The destination to store the correlation values in. The result is normed.
 */
void autocov(const cv::Mat &img, cv::Mat &dst)
{
	//Convert image from unsigned char to float matrix
	cv::Mat fImg;
	img.convertTo(fImg, CV_32FC1);
	//Subtract the mean
	cv::Mat mean(fImg.size(), fImg.type(), cv::mean(fImg));
	cv::subtract(fImg, mean, fImg);

	dst = cv::Mat(fImg.size(), CV_32FC1, cv::Scalar::all(0));

	for (int x = - img.cols/2; x < img.cols/2; x++)
	{
		for (int y = -img.rows/2; y < img.rows/2; y++)
		{
			int from_i = x < 0 ? 0 - x : 0;
			int from_j = y < 0 ? 0 - y : 0;
			int to_i   = x < 0 ? img.cols : img.cols - x;
			int to_j   = y < 0 ? img.rows : img.rows - y;
			for (int i = from_i; i < to_i; i++)
			{	
				for (int j = from_j; j < to_j; j++)
				{	
					dst.at<float>(y+img.rows/2,x+img.cols/2) += fImg.at<float>(j, i) * fImg.at<float>(j+y, i+x);
					dst.at<float>(y+img.rows/2,x+img.cols/2) /= 1.0 * (fImg.cols - abs(x)) * (fImg.rows - abs(y));
				}
			}
		}
	}	
	//norm the result
	cv::multiply(fImg,fImg,fImg);
	float denom = cv::sum(fImg)[0];
	dst = dst * (1/(denom * 1.0/(img.rows*img.cols)));
}

int main (int argc, char** argv)
{

	if (argc != 3)
	{
		cout<<"Usage: "<<argv[0]<<" <filename> <minimal pattern size>"<<endl;
		return EXIT_FAILURE;	
	}
	else
	{
		cv::Mat src = cv::imread(argv[1], 0);

		//try to find a pattern in the image and get the pattern size
		lssr::AutoCorr* ac = new lssr::AutoCorr(src);
		unsigned int sizeX, sizeY, sX, sY;
		cout<<"confidence: "<<ac->getMinimalPattern(sX, sY, sizeX, sizeY, atoi(argv[2]))<<endl;

		//save the pattern
		cv::Mat pattern = cv::Mat(src, cv::Rect(sX, sY, sizeX, sizeY));

		//repeat the pattern to visualize the result
		cv::Mat repeatedPattern = cv::repeat(pattern, 3, 3);

		cv::imwrite("p1.jpg", pattern);

		cv::startWindowThread();
		
		//show the single pattern
		cv::namedWindow("Pattern", CV_WINDOW_AUTOSIZE);
		cv::imshow("Pattern", pattern);
		cv::waitKey();
		
		//show the stitched pattern
		cv::namedWindow("RepeatedPattern", CV_WINDOW_AUTOSIZE);
		cv::imshow("RepeatedPattern", repeatedPattern);
		cv::waitKey();

		cv::destroyAllWindows();

		return EXIT_SUCCESS;
	}

}
