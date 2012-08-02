/* Copyright (C) 2011 Uni Osnabrück
 * This file is part of the LAS VEGAS Reconstruction Toolkit,
 *
 * LAS VEGAS is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * LAS VEGAS is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA
 */


/*
 * AutoCorr.cpp
 *
 *  @date 15.07.2012
 *  @author Kim Rinnewitz (krinnewitz@uos.de)
 */

#include "AutoCorr.hpp"

namespace lssr {
AutoCorr::AutoCorr(Texture *t)
{
	//convert texture to cv::Mat
	cv::Mat img1(cv::Size(t->m_width, t->m_height), CV_MAKETYPE(t->m_numBytesPerChan * 8, t->m_numChannels), t->m_data);

	//make input gray scale 
	cv::Mat img;	
	cv::cvtColor(img1, img, CV_RGB2GRAY);

	m_image = img1;
	autocorrDFT(img, m_autocorr);
	
}

AutoCorr::AutoCorr(const cv::Mat &t)
{
	m_image = t;
	autocorrDFT(t, m_autocorr);
}

void AutoCorr::autocorrDFT(const cv::Mat &img, cv::Mat &dst)
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

void AutoCorr::getACX(const cv::Mat &ac, float* &output)
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

void AutoCorr::getACY(const cv::Mat &ac, float* &output)
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

float AutoCorr::calcStdDev(const int* data, int len)
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

int AutoCorr::countPeaks(const float* data, float &stdDev, int len, int peaks[])
{
	const float epsilon = 0.0001;
	int result = 0;

	if (len < 2)
	{
		return 0;
	}

	int lastPeak = -1;

	bool curr_up = true;

	//Count borders, too
	if (data[0] > data[1])
	{
		peaks[result] = 0;
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
			peaks[result] = i - 1;
			result++;
			lastPeak = i - 1;
		}
		curr_up = next_up;
	}

	if (data[len-1] > data[len-2])
	{
		if (lastPeak != -1)
		{
			distances[result-1] = lastPeak - (len - 1);
		}
		peaks[result] = len - 1;
		result++;
	}

	stdDev = calcStdDev(distances, result - 1);

	return result;
}

float AutoCorr::calcPeakHeight(int peak, float* data, int len)
{
	float result = 0;

	//find deeps before and after peak
	int d1 = peak;
	while(data[d1] >= data[d1 - 1] && d1 > 0)
	{
		d1--;
	}
	int d2 = peak;
	while(data[d2] >= data[d2 + 1] && d2 < len - 1)
	{
		d2++;
	}

	//calculate height of peak
	result = data[peak] - (data[d1] + data[d2]) / 2;

	return result;	
}

double AutoCorr::getMinimalPatternCC(unsigned int &sX, unsigned int &sY, unsigned int &sizeX, unsigned int &sizeY, const int minimalPatternSize = 10)
{
	const float epsilon = 0.00005;

	float *rho_x = 0;
	float *rho_y = 0;	
	getACX(m_autocorr, rho_x);
	getACY(m_autocorr, rho_y);
	float stdDevX = 0;
	float stdDevY = 0;
	int* xPeaks = new int[m_autocorr.cols];
	int* yPeaks = new int[m_autocorr.rows];
	int peaksX = countPeaks(rho_x, stdDevX, m_autocorr.cols, xPeaks);
	int peaksY = countPeaks(rho_y, stdDevY, m_autocorr.rows, yPeaks);
//	std::cout<<"Peaks x:"<<peaksX<<"\t\t StdDev rho_x: "<<stdDevX/(m_autocorr.cols / peaksX)<<std::endl;
//	std::cout<<"Peaks y:"<<peaksY<<"\t\t StdDev rho_y: "<<stdDevY/(m_autocorr.rows / peaksY)<<std::endl;
	for (int i = 0; i < m_autocorr.cols; i++)
	{
//		std::cerr<<i<<" "<<rho_x[i]<<std::endl;
	}
//	for (int i = 0; i < peaksX; i++)
//	{
//		std::cout<<xPeaks[i]<<" "<<0<<std::endl;
//	}

	std::vector<int> high_xPeaks;
	float meanPeakHeight = 0;
	for (int i = 0; i < peaksX; i++)
	{
		meanPeakHeight += calcPeakHeight(xPeaks[i], rho_x, m_autocorr.cols);
	}
	meanPeakHeight /= peaksX;
	for (int i = 0; i < peaksX; i++)
	{
		if (calcPeakHeight(xPeaks[i], rho_x, m_autocorr.cols) > meanPeakHeight)// minimalPatternSize && rho_x[xPeaks[i]] > 0)
		{
			high_xPeaks.push_back(xPeaks[i]);
	//		std::cout<<high_xPeaks.back()<<" "<<0<<std::endl;
		}
	}
	std::vector<int> high_yPeaks;
	meanPeakHeight = 0;
	for (int i = 0; i < peaksY; i++)
	{
		meanPeakHeight += calcPeakHeight(yPeaks[i], rho_y, m_autocorr.rows);
	}
	meanPeakHeight /= peaksY;
	for (int i = 0; i < peaksY; i++)
	{
		if (calcPeakHeight(yPeaks[i], rho_y, m_autocorr.rows) > meanPeakHeight)//minimalPatternSize && rho_y[yPeaks[i]] > 0)
		{
			high_yPeaks.push_back(yPeaks[i]);
		}
	}

	//x direction
	float x_highest_correlation = -FLT_MAX;
	for (int x = 0; x < high_xPeaks.size() / 2; x++)
	{
		for (int x2 = x + 1; x2 < high_xPeaks.size(); x2++)
		{
			//choose size for subrect
			int width  = high_xPeaks[x2] - high_xPeaks[x];
			int height = m_image.rows;

			//choose center for the subrect
			int cx = high_xPeaks[x] + width / 2;
			int cy = height / 2;

			//extract the subrect
			cv::Mat dst;
			cv::getRectSubPix(m_image, cv::Size(width, height), cv::Point2f(cx, cy), dst);
		
			//calculate cross correlation between dst and m_image
			CrossCorr* cc = new CrossCorr(m_image, dst);
			float correlation = 0;	
			for (int i = 0; i < m_image.cols / width; i++)
			{
				correlation += fabs(cc->at((high_xPeaks[x] + i * width) % m_image.cols, 0));
			}	
//			std::cout<<high_xPeaks[x]<<" "<<high_xPeaks[x2]<<" "<<correlation<<std::endl;
			if (correlation > x_highest_correlation)
			{
				x_highest_correlation = correlation;
				sizeX 	= width;
				sX	= high_xPeaks[x];
			}
		}
	}
	//Could not find enough peaks
	if (x_highest_correlation == -FLT_MAX)
	{
 		sizeX 	= m_image.cols;
		sX	= 0;
	}

	//y direction
	float y_highest_correlation = -FLT_MAX;
	for (int y = 0; y < high_yPeaks.size(); y++)
	{
		for (int y2 = y + 1; y2 < high_yPeaks.size() / 2; y2++)
		{
			//choose size for subrect
			int width = sizeX;
			int height  = high_yPeaks[y2] - high_yPeaks[y];

			//choose center for the subrect
			int cx = sX;
			int cy = high_yPeaks[y] + width / 2;

			//extract the subrect
			cv::Mat dst;
			cv::getRectSubPix(m_image, cv::Size(width, height), cv::Point2f(cx, cy), dst);
		
			//calculate cross correlation between dst and m_image
			CrossCorr* cc = new CrossCorr(m_image, dst);
			float correlation = 0;	
			for (int i = 0; i < m_image.rows / height; i++)
			{
				for (int j = 0; j < m_image.cols / width; j++)
				{
					correlation += fabs(cc->at((sX + j * width) % m_image.cols, (high_yPeaks[y] + i * height) % m_image.rows));
				}
			}	
	//		std::cout<<std::endl<<high_yPeaks[y]<<" "<<high_yPeaks[y2]<<" "<<correlation<<std::endl;
			if (correlation > y_highest_correlation)
			{
				y_highest_correlation = correlation;
				sizeY 	= height;
				sY	= high_yPeaks[y];
			}
		}
	}
	//Could not find enough peaks
	if (y_highest_correlation == -FLT_MAX)
	{
 		sizeY 	= m_image.rows;
		sY	= 0;
	}
	
	if (y_highest_correlation == -FLT_MAX == x_highest_correlation)
	{
		//Texture is aperiodic
		return 0;
	}
	else
	{
		return	1.0f / (stdDevX/(m_autocorr.cols / peaksX) + stdDevY/(m_autocorr.rows / peaksY));
	}
} 

class patternDim{
public:
	unsigned int s;
	unsigned int size;
	float fit;
	bool operator()(patternDim a, patternDim b)
	{
		return a.fit < b.fit;
	}
};


double AutoCorr::getMinimalPattern(unsigned int &sX, unsigned int &sY, unsigned int &sizeX, unsigned int &sizeY, const int minimalPatternSize = 10)
{
	const float epsilon = 0.00005;

	float *rho_x = 0;
	float *rho_y = 0;	
	getACX(m_autocorr, rho_x);
	getACY(m_autocorr, rho_y);
	float stdDevX = 0;
	float stdDevY = 0;
	int* xPeaks = new int[m_autocorr.cols];
	int* yPeaks = new int[m_autocorr.rows];
	int peaksX = countPeaks(rho_x, stdDevX, m_autocorr.cols, xPeaks);
	int peaksY = countPeaks(rho_y, stdDevY, m_autocorr.rows, yPeaks);
//	std::cout<<"Peaks x:"<<peaksX<<"\t\t StdDev rho_x: "<<stdDevX/(m_autocorr.cols / peaksX)<<std::endl;
//	std::cout<<"Peaks y:"<<peaksY<<"\t\t StdDev rho_y: "<<stdDevY/(m_autocorr.rows / peaksY)<<std::endl;
	for (int i = 0; i < m_autocorr.cols; i++)
	{
		std::cerr<<i<<" "<<rho_x[i]<<std::endl;
	}
//	for (int i = 0; i < peaksX; i++)
//	{
//		std::cout<<xPeaks[i]<<" "<<0<<std::endl;
//	}

	std::vector<int> high_xPeaks;
	for (int i = 0; i < peaksX; i++)
	{
		if (/*calcPeakHeight(xPeaks[i], rho_x, m_autocorr.cols) > minimalPatternSize &&*/ rho_x[xPeaks[i]] > 0)
		{
			high_xPeaks.push_back(xPeaks[i]);
//			std::cout<<high_xPeaks.back()<<" "<<0<<std::endl;
		}
	}
	std::vector<int> high_yPeaks;
	for (int i = 0; i < peaksY; i++)
	{
		if (/*calcPeakHeight(yPeaks[i], rho_y, m_autocorr.rows) > minimalPatternSize &&*/ rho_y[yPeaks[i]] > 0)
		{
			high_yPeaks.push_back(yPeaks[i]);
		}
	}

	std::priority_queue<patternDim, std::vector<patternDim>, patternDim> x_patterns;
	//x direction
	for (int x = 0; x < high_xPeaks.size() / 2; x++)
	{
		for (int x2 = x + 1; x2 < high_xPeaks.size() / 2; x2++)
		{
			patternDim p;	
			float dist = fabs(calcPeakHeight(high_xPeaks[x], rho_x, m_autocorr.cols) - calcPeakHeight(high_xPeaks[x2], rho_x, m_autocorr.cols));
			float d1   = fabs(fabs(rho_x[high_xPeaks[x]] - rho_x[high_xPeaks[x] - 1]) - fabs(rho_x[high_xPeaks[x2]] - rho_x[high_xPeaks[x2] - 1]));
			float d2   = fabs(fabs(rho_x[high_xPeaks[x]] - rho_x[high_xPeaks[x] + 1]) - fabs(rho_x[high_xPeaks[x2]] - rho_x[high_xPeaks[x2] + 1]));
			p.size 	= high_xPeaks[x2] - high_xPeaks[x];
			p.s	= high_xPeaks[x];
			p.fit = dist + d1 + d2;
			x_patterns.push(p);
		}
	}
	
	if (x_patterns.empty())
	{
		sizeX = m_image.cols;
		sX = 0;
	}
	else
	{
		//Test the first elements of the priority queuer by cross correlation
		float x_highest_correlation = -FLT_MAX;
		for (int x = 0; x < std::min((size_t)10, x_patterns.size()); x++)
		{
			//choose size for subrect
			int width  = x_patterns.top().size;
			int height = m_image.rows;

			//choose center for the subrect
			int cx = x_patterns.top().s + width / 2;
			int cy = height / 2;

			//extract the subrect
			cv::Mat dst;
			cv::getRectSubPix(m_image, cv::Size(width, height), cv::Point2f(cx, cy), dst);
		
			//calculate cross correlation between dst and m_image
			CrossCorr* cc = new CrossCorr(m_image, dst);
			float correlation = 0;	
			for (int i = 0; i < m_image.cols / width; i++)
			{
				correlation += cc->at((x_patterns.top().s + i * width) % m_image.cols, 0);
			}	
//			std::cout<<"x: "<<x<<" "<<correlation<<std::endl;
			if (correlation > x_highest_correlation)
			{
				x_highest_correlation = correlation;
				sizeX 	= width;
				sX	= x_patterns.top().s;
			}
			x_patterns.pop();
		}
	}

	//y direction
	std::priority_queue<patternDim, std::vector<patternDim>, patternDim> y_patterns;
	for (int y = 0; y < high_yPeaks.size() / 2; y++)
	{
		for (int y2 = y + 1; y2 < high_yPeaks.size() / 2; y2++)
		{
			patternDim p;	
			float dist = fabs(calcPeakHeight(high_yPeaks[y], rho_y, m_autocorr.rows) - calcPeakHeight(high_yPeaks[y2], rho_y, m_autocorr.rows));
			float d1   = fabs(fabs(rho_y[high_yPeaks[y]] - rho_y[high_yPeaks[y] - 1]) - fabs(rho_y[high_yPeaks[y2]] - rho_y[high_yPeaks[y2] - 1]));
			float d2   = fabs(fabs(rho_y[high_yPeaks[y]] - rho_y[high_yPeaks[y] + 1]) - fabs(rho_y[high_yPeaks[y2]] - rho_y[high_yPeaks[y2] + 1]));
			p.size 	= high_yPeaks[y2] - high_yPeaks[y];
			p.s	= high_yPeaks[y];
			p.fit = dist + d1 + d2;
			y_patterns.push(p);
		}
	}

	//Could not find more than one peak
	if (y_patterns.empty())
	{
		sizeY = m_image.rows;
		sY = 0;
	}
	else
	{
		//Test the first elements of the priority queuer by cross correlation
		float y_highest_correlation = -FLT_MAX;
		for (int y = 0; y < std::min((size_t)10, y_patterns.size()); y++)
		{
			//choose size for subrect
			int width = m_image.cols;
			int height  = y_patterns.top().size;

			//choose center for the subrect
			int cx = width / 2;
			int cy = y_patterns.top().s + width / 2;

			//extract the subrect
			cv::Mat dst;
			cv::getRectSubPix(m_image, cv::Size(width, height), cv::Point2f(cx, cy), dst);
		
			//calculate cross correlation between dst and m_image
			CrossCorr* cc = new CrossCorr(m_image, dst);
			float correlation = 0;	
			for (int i = 0; i < m_image.rows / height; i++)
			{
				correlation += cc->at(0, (y_patterns.top().s + i * height) % m_image.rows);
			}	
//			std::cout<<"y: "<<y<<" "<<correlation<<std::endl;
			if (correlation > y_highest_correlation)
			{
				y_highest_correlation = correlation;
				sizeY 	= height;
				sY	= y_patterns.top().s;
			}
			y_patterns.pop();
		}
	}

	return 0; //TODO
} 

AutoCorr::~AutoCorr()
{
	//TODO?!
}
}
