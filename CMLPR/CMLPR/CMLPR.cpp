// CMLPR.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include "core/core.hpp"
#include "highgui/highgui.hpp"
#include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;

Mat RGB2Grey(Mat RGB)
{
	Mat grey = Mat::zeros(RGB.size(), CV_8UC1);

	for (int i = 0; i < RGB.rows; i++)
	{
		for (int j = 0; j < RGB.cols * 3; j += 3)
		{
			grey.at<uchar>(i, j / 3) = (RGB.at<uchar>(i, j) + RGB.at<uchar>(i, j + 1) + RGB.at<uchar>(i, j + 2)) / 3;
		}
	}

	return grey;
}


Mat Grey2Binary(Mat Grey, int treshold)
{
	Mat bin = Mat::zeros(Grey.size(), CV_8UC1);

	for (int i = 0; i < Grey.rows; i++)
	{
		for (int j = 0; j < Grey.cols; j+=3)
		{
			if (Grey.at<uchar>(i, j) > treshold)
				bin.at<uchar>(i, j) = 255;

		}
	}
	return bin;
}

Mat Inversion(Mat Grey)
{
	Mat invertedImg = Mat::zeros(Grey.size(), CV_8UC1);

	for (int i = 0; i < Grey.rows; i++)
	{
		for (int j = 0; j < Grey.cols; j++)
		{
			invertedImg.at<uchar>(i, j) = 255 - Grey.at<uchar>(i, j);

		}
	}
	return invertedImg;
}

Mat Step(Mat Grey, int th1, int th2)
{
	Mat output = Mat::zeros(Grey.size(), CV_8UC1);

	for (int i = 0; i < Grey.rows; i++)
	{
		for (int j = 0; j < Grey.cols; j++)
		{
			if (Grey.at<uchar>(i, j) >= th1 && Grey.at<uchar>(i, j) <= th2)
				output.at<uchar>(i, j) = 255;
		}
	}
	return output;
}

Mat Average(Mat Grey, int neighbirSize)
{
	Mat AvgImg = Mat::zeros(Grey.size(), CV_8UC1);
	int totalPix = pow(2 * neighbirSize + 1, 2);
	for (int i = neighbirSize; i < Grey.rows - neighbirSize; i++)
	{
		for (int j = neighbirSize; j < Grey.cols - neighbirSize; j++)
		{
			int sum = 0;
			int count = 0;
			for (int ii = -neighbirSize; ii <= neighbirSize; ii++)
			{
				for (int jj = -neighbirSize; jj <= neighbirSize; jj++)
				{
					count++;
					sum += Grey.at<uchar>(i + ii, j + jj);
				}
			}
			AvgImg.at<uchar>(i, j) = sum / count;
			//AvgImg.at<uchar>(i, j) = (Grey.at<uchar>(i-1, j-1) + Grey.at<uchar>(i - 1, j ) + Grey.at<uchar>(i - 1, j + 1)+ Grey.at<uchar>(i , j - 1) + Grey.at<uchar>(i , j ) + Grey.at<uchar>(i, j + 1) + Grey.at<uchar>(i+1, j - 1) + Grey.at<uchar>(i+1, j) + Grey.at<uchar>(i+1, j + 1))/9;

		}
	}

	return AvgImg;
}


Mat Max(Mat Grey, int neighbirSize)
{
	Mat MaxImg = Mat::zeros(Grey.size(), CV_8UC1);
	for (int i = neighbirSize; i < Grey.rows - neighbirSize; i++)
	{
		for (int j = neighbirSize; j < Grey.cols - neighbirSize; j++)
		{
			int Defval = -1;
			for (int ii = -neighbirSize; ii <= neighbirSize; ii++)
			{
				for (int jj = -neighbirSize; jj <= neighbirSize; jj++)
				{

					if (Grey.at<uchar>(i + ii, j + jj) > Defval)
						Defval = Grey.at<uchar>(i + ii, j + jj);
				}
			}
			MaxImg.at<uchar>(i, j) = Defval;
			//AvgImg.at<uchar>(i, j) = (Grey.at<uchar>(i-1, j-1) + Grey.at<uchar>(i - 1, j ) + Grey.at<uchar>(i - 1, j + 1)+ Grey.at<uchar>(i , j - 1) + Grey.at<uchar>(i , j ) + Grey.at<uchar>(i, j + 1) + Grey.at<uchar>(i+1, j - 1) + Grey.at<uchar>(i+1, j) + Grey.at<uchar>(i+1, j + 1))/9;

		}
	}

	return MaxImg;
}

Mat Min(Mat Grey, int neighbirSize)
{
	Mat MinImg = Mat::zeros(Grey.size(), CV_8UC1);
	for (int i = neighbirSize; i < Grey.rows - neighbirSize; i++)
	{
		for (int j = neighbirSize; j < Grey.cols - neighbirSize; j++)
		{
			int Defval = 255;
			for (int ii = -neighbirSize; ii <= neighbirSize; ii++)
			{
				for (int jj = -neighbirSize; jj <= neighbirSize; jj++)
				{

					if (Grey.at<uchar>(i + ii, j + jj) < Defval)
						Defval = Grey.at<uchar>(i + ii, j + jj);
				}
			}
			MinImg.at<uchar>(i, j) = Defval;
			//AvgImg.at<uchar>(i, j) = (Grey.at<uchar>(i-1, j-1) + Grey.at<uchar>(i - 1, j ) + Grey.at<uchar>(i - 1, j + 1)+ Grey.at<uchar>(i , j - 1) + Grey.at<uchar>(i , j ) + Grey.at<uchar>(i, j + 1) + Grey.at<uchar>(i+1, j - 1) + Grey.at<uchar>(i+1, j) + Grey.at<uchar>(i+1, j + 1))/9;

		}
	}

	return MinImg;
}

Mat Edge(Mat Grey, int th)
{
	Mat EdgeImg = Mat::zeros(Grey.size(), CV_8UC1);
	for (int i = 1; i < Grey.rows - 1; i++)
	{
		for (int j = 1; j < Grey.cols - 1; j++)
		{
			int AvgL = (Grey.at<uchar>(i - 1, j - 1) + Grey.at<uchar>(i, j - 1) + Grey.at<uchar>(i + 1, j - 1)) / 3;
			int AvgR = (Grey.at<uchar>(i - 1, j + 1) + Grey.at<uchar>(i, j + 1) + Grey.at<uchar>(i + 1, j + 1)) / 3;
			if (abs(AvgL - AvgR) > th)
				EdgeImg.at<uchar>(i, j) = 255;


		}
	}
	return EdgeImg;
}

Mat Dilation(Mat EdgeImg, int neighbirSize)
{
	Mat DilatedImg = Mat::zeros(EdgeImg.size(), CV_8UC1);
	for (int i = neighbirSize; i < EdgeImg.rows - neighbirSize; i++)
	{
		for (int j = neighbirSize; j < EdgeImg.cols - neighbirSize; j++)
		{
			for (int ii = -neighbirSize; ii <= neighbirSize; ii++)
			{
				for (int jj = -neighbirSize; jj <= neighbirSize; jj++)
				{
					if (EdgeImg.at<uchar>(i, j) == 0)
					{
						if (EdgeImg.at<uchar>(i + ii, j + jj) == 255)
						{
							DilatedImg.at<uchar>(i, j) = 255;
							break;
						}
					}
					else
						DilatedImg.at<uchar>(i, j) = 255;



				}
			}
			//AvgImg.at<uchar>(i, j) = (Grey.at<uchar>(i-1, j-1) + Grey.at<uchar>(i - 1, j ) + Grey.at<uchar>(i - 1, j + 1)+ Grey.at<uchar>(i , j - 1) + Grey.at<uchar>(i , j ) + Grey.at<uchar>(i, j + 1) + Grey.at<uchar>(i+1, j - 1) + Grey.at<uchar>(i+1, j) + Grey.at<uchar>(i+1, j + 1))/9;
		}
	}

	return DilatedImg;
}

Mat Erosion(Mat Edge, int windowsize) {
	Mat ErodedImg = Mat::zeros(Edge.size(), CV_8UC1);
	for (int i = windowsize; i < Edge.rows - windowsize; i++) {
		for (int j = windowsize; j < Edge.cols - windowsize; j++) {
			ErodedImg.at<uchar>(i, j) = Edge.at<uchar>(i, j);
			for (int p = -windowsize; p <= windowsize; p++) {
				for (int q = -windowsize; q <= windowsize; q++) {
					if (Edge.at<uchar>(i + p, j + q) == 0) {
						ErodedImg.at<uchar>(i, j) = 0;



					}
				}
			}
		}
	}
	return ErodedImg;
}


int main()
{
		Mat img;
		img = imread("C:\\Users\\jason\\source\\repos\\CMLPR\\Images\\0.png");
		//imshow("RBG image" , img);

		Mat GreyImg = RGB2Grey(img);
		//imshow("Grey image", GreyImg);

		Mat BlurredImg = Average(GreyImg, 1);
		//imshow("Blurred image", BlurredImg);

		Mat EdgeImg = Edge(BlurredImg, 50);
		//imshow("Edge image", EdgeImg);

		Mat ErosionImg = Erosion(EdgeImg, 1);
		//imshow("Edge image", EdgeImg);

		Mat DialatedImg = Dilation(ErosionImg, 15);
		//imshow("Dialated image", DialatedImg);

		Mat DilatedImgCpy;
		DilatedImgCpy = DialatedImg.clone();
		vector<vector<Point>> contours1;
		vector<Vec4i> hierachy1;
		findContours(DialatedImg, contours1, hierachy1, RETR_EXTERNAL, CHAIN_APPROX_NONE, Point(0, 0));
		Mat dst = Mat::zeros(GreyImg.size(), CV_8UC3);

		if (!contours1.empty())
		{
			for (int i = 0; i < contours1.size(); i++)
			{
				Scalar colour((rand() & 255), (rand() & 255), (rand() & 255));
				drawContours(dst, contours1, i, colour, -1, 8, hierachy1);
			}
		}

		Mat plate;
		Rect rect;

		Scalar black = CV_RGB(0, 0, 0);
		for (int i = 0; i < contours1.size(); i++)
		{
			rect = boundingRect(contours1[i]);

			bool ToSmall = rect.width < 60 || rect.height < 20;
			bool ToBig = rect.width > 200;  
			bool OutOfBounds = rect.x < 0.1 * GreyImg.cols || rect.x > 0.9 * GreyImg.cols || rect.y < 0.1 * GreyImg.rows || rect.y > 0.9 * GreyImg.rows;


			float ratio = ((float)rect.width / (float)rect.height);
			if (ToSmall || ToBig || OutOfBounds || ratio < 1.5)
			{
				drawContours(DilatedImgCpy, contours1, i, black, -1, 8, hierachy1);
			}
			else
				plate = GreyImg(rect);

		}

		imshow("Filtered Image", DilatedImgCpy);
		if (plate.rows != 0 && plate.cols != 0)
			imshow("Detected Plate", plate);
	
		std::cout << img.rows << "x" << img.cols;

	

    waitKey();
}
