#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <opencv/cv.h>
#include <opencv/highgui.h>
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
double autocorr(int y, int x, cv::Mat &img)
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

/**
 * \brief 	Implementation of the auto correlation function using fourier transformation.
 *		This implementation is quite fast and may be used for productive jobs. Auto
 *		correlation can be calculated by transforming the image img into the frequency
 *		domain (getting the fourier transformation IMG of img), calculating
 *		IMG * IMG  and transforming the result back to the image domain. 
 *
 * \param 	img	The image to calculate the auto correlation for. Must be one channel
 *			gray scale.
 * \param	dst	The destination to store the correlation values in. The result is normed.
 */
void autocorrDFT(const cv::Mat &img, cv::Mat &dst)
{
	//Convert image from unsigned char to float matrix
	cv::Mat fImg;
	img.convertTo(fImg, CV_32FC1);
	//Subtract the mean
	cv::Mat mean(fImg.size(), fImg.type(), cv::mean(fImg));
	cv::subtract(fImg, mean, fImg);
	
	//Calculate the optimal size for the dft output.
	//This increases speed.
	cv::Size dftSize;
	dftSize.width = cv::getOptimalDFTSize(2 * img.cols +1 );
	dftSize.height = cv::getOptimalDFTSize(2 * img.rows +1);
	
	//prepare the destination for the dft
	dst = cv::Mat(dftSize, CV_32FC1, cv::Scalar::all(0));
	
	//transform the image into the frequency domain
	cv::dft(fImg, dst);
	//calculate DST * DST (don't mind the fourth parameter. It is ignored)
	cv::mulSpectrums(dst, dst, dst, cv::DFT_INVERSE, true);
	//transform the result back to the image domain 
	cv::dft(dst, dst, cv::DFT_INVERSE | cv::DFT_SCALE);

	//norm the result
	cv::multiply(fImg,fImg,fImg);
	float denom = cv::sum(fImg)[0];
	dst = dst * (1/denom);

}

/**
 * \brief	Calculates the sum of all rows for each column of
 *		the given autocorrelation matrix. It will return
 *		a float array of length ac.cols.
 *
 * \param	ac	The auto correlation matrix
 * \param	output	The destination where results are stored
 */
void getACX(const cv::Mat &ac, float* &output)
{
	//Allocate output
	output = new float[ac.cols];
	for(int x = 0; x < ac.cols; x++)
	{
		float rho_x = 0;
		for(int y = 0; y < ac.rows; y++)
		{
			rho_x += ac.at<float>(y,x);
		}
		output[x] = rho_x;
	}
}

/**
 * \brief	Calculates the sum of all columns for each row of
 *		the given autocorrelation matrix. It will return
 *		a float array of length ac.rows.
 *
 * \param	ac	The auto correlation matrix
 * \param	output	The destination where results are stored
 */
void getACY(const cv::Mat &ac, float* &output)
{
	//Allocate output
	output = new float[ac.rows];
	for(int y = 0; y < ac.rows; y++)
	{
		float rho_y = 0;
		for(int x = 0; x < ac.cols; x++)
		{
			rho_y += ac.at<float>(y,x);
		}
		output[y] = rho_y;
	}
}

/**
 * \brief 	Calculates the standard deviation of the given 
 *		data array
 *
 * \param	data	The data array
 * \param	len 	The length of the data array
 *
 * \return 	The standard deviation of the given data array
 */
float calcStdDev(const int* data, int len)
{
	float result = 0;
	float mean = 0;
	for (int i = 0; i < len; i++)
	{
		mean += data[i];
	}	
	mean /= len;

	for (int i = 0; i < len; i++)
	{
		result += (data[i] - mean) * (data[i] - mean);
	}
	result /= len;
	result = sqrt(result);

	return result;
}

/**
 * \brief 	Counts the number of peaks in the given data Array and
 * 		returns the standard deviation of the distances between
 *		the peaks
 * \param	data		The data array
 * \param	stdDev		The standard deviation of the distances
 *				between the peaks
 * \param	len		The length of the array
 *
 * \return	The number of peaks in the data array
 */
int countPeaks(const float* data, float &stdDev, int len)
{
	const float epsilon = 0.0001;
	int result = 0;

	if (len < 2)
	{
		return 0;
	}

	int lastPeak = -1;

	bool curr_up = true;

	//Count boarders, too
	if (data[0] > data[1])
	{
		result++;
		lastPeak = 0;
		curr_up = false;
	}

	int* distances = new int[len];

	//Search for peaks
	for (int i = 1; i < len - 1; i++)
	{
		bool next_up = curr_up;
//		if (data[i] > data[i-1])  
		if (data[i] - data[i-1] > epsilon)
		{
			next_up = true;
		}
//		if (data[i] < data[i-1])  
		if (data[i] - data[i-1] < -epsilon)  
		{
			next_up = false;
		}
		if (next_up == false && curr_up == true)
		{
			//peak detected
			if (lastPeak != -1)
			{
				distances[result-1] = lastPeak - i;
			}
			result++;
			lastPeak = i;
		}
		curr_up = next_up;
	}

	if (data[len-1] > data[len-2])
	{
		if (lastPeak != -1)
		{
			distances[result-1] = lastPeak - (len - 1);
		}
		result++;
	}

	stdDev = calcStdDev(distances, result - 1);

	return result;
}


/**
 * \brief	Tries to find a pattern in an Image using the auto correlation
 *		function. The result can be interpreted as a rectangle at the
 *		origin (0,0) of the input image with the width of sizeX and the
 * 		height of sizeY.
 *
 * \param	input			The image to find a pattern in. Has to be
					a three channel	color (RGB) image.
 * \param	sizeX			The resulting x size of the found pattern
 * \param	sizeY			The resulting y size of the found pattern
 * \param	minimalPatternSize	The minimum acceptable x and y size of a
 *					pattern 
 *
 * \return	A confidence between 0 and 1 indicating the degree of success in
 *		extracting a pattern from the given image
 */
double getMinimalPattern(const cv::Mat &input, unsigned int &sizeX, unsigned int &sizeY, const int minimalPatternSize = 10)
{
	const float epsilon = 0.00005;

	//make input gray scale 
	cv::Mat img;	
	cv::cvtColor(input, img, CV_RGB2GRAY);

	//calculate auto correlation
	cv::Mat ac;
	autocorrDFT(img, ac);
	cv::Mat_<float>& ptrAc = (cv::Mat_<float>&)ac;

//===========================
	float *rho_x = 0;
	float *rho_y = 0;	
	getACX(ac, rho_x);
	getACY(ac, rho_y);
	float stdDevX = 0;
	float stdDevY = 0;
	int peaksX = countPeaks(rho_x, stdDevX, ac.cols);
	int peaksY = countPeaks(rho_y, stdDevY, ac.rows);
	cout<<"Peaks x:"<<peaksX<<"\t\t StdDev rho_x: "<<stdDevX/(ac.cols / peaksX)<<endl;
	cout<<"Peaks y:"<<peaksY<<"\t\t StdDev rho_y: "<<stdDevY/(ac.rows / peaksY)<<endl;
//==========================

	//search minimal pattern i.e. search the highest correlation in x and y direction
	sizeX = 0;
	sizeY = 0;

	//y direction
/*	for (int y = minimalPatternSize; y < ac.size().height / 2; y++)
	{
		for(int x = 1; x < ac.cols / 2; x++)
		{
			if (ptrAc(y, x) > ptrAc(sizeY, sizeX) + epsilon || sizeY == 0)
			{
				sizeY = y;	
				sizeX = x;
			}
		}
	}

	sizeX = 0;
	
	//x direction
	for (int x = minimalPatternSize; x < ac.size().width / 2; x++)
	{
		if (ptrAc(sizeY, x) > ptrAc(sizeY, sizeX) + epsilon || sizeX == 0)
		{
			sizeX = x;	
		}
	}
*/	
	sizeX = 1; sizeY = 1; //TODO: remove
	return ptrAc(sizeY, sizeX);
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
		cv::Mat src = cv::imread(argv[1]);

		//try to find a pattern in the image and get the pattern size
		unsigned int sizeX, sizeY;
		cout<<"confidence: "<<getMinimalPattern(src, sizeX, sizeY, atoi(argv[2]))<<endl;

		//save the pattern
		cv::Mat pattern = cv::Mat(src, cv::Rect(0, 0, sizeX, sizeY));

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
