// OpenCVApplication.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "common.h"
#include <queue>
#include <random>
#include <stack>

void testOpenImage()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		imshow("image",src);
		waitKey();
	}
}

void testOpenImagesFld()
{
	char folderName[MAX_PATH];
	if (openFolderDlg(folderName)==0)
		return;
	char fname[MAX_PATH];
	FileGetter fg(folderName,"bmp");
	while(fg.getNextAbsFile(fname))
	{
		Mat src;
		src = imread(fname);
		imshow(fg.getFoundFileName(),src);
		if (waitKey()==27) //ESC pressed
			break;
	}
}

void testImageOpenAndSave()
{
	Mat src, dst;

	src = imread("Images/Lena_24bits.bmp", IMREAD_COLOR);	// Read the image

	if (!src.data)	// Check for invalid input
	{
		printf("Could not open or find the image\n");
		return;
	}

	// Get the image resolution
	Size src_size = Size(src.cols, src.rows);

	// Display window
	const char* WIN_SRC = "Src"; //window for the source image
	namedWindow(WIN_SRC, WINDOW_AUTOSIZE);
	moveWindow(WIN_SRC, 0, 0);

	const char* WIN_DST = "Dst"; //window for the destination (processed) image
	namedWindow(WIN_DST, WINDOW_AUTOSIZE);
	moveWindow(WIN_DST, src_size.width + 10, 0);

	cvtColor(src, dst, COLOR_BGR2GRAY); //converts the source image to a grayscale one

	imwrite("Images/Lena_24bits_gray.bmp", dst); //writes the destination to file

	imshow(WIN_SRC, src);
	imshow(WIN_DST, dst);

	printf("Press any key to continue ...\n");
	waitKey(0);
}

void testNegativeImage()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		double t = (double)getTickCount(); // Get the current time [s]
		
		Mat src = imread(fname,IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = Mat(height,width,CV_8UC1);
		// Asa se acceseaaza pixelii individuali pt. o imagine cu 8 biti/pixel
		// Varianta ineficienta (lenta)
		for (int i=0; i<height; i++)
		{
			for (int j=0; j<width; j++)
			{
				uchar val = src.at<uchar>(i,j);
				uchar neg = 255 - val;
				dst.at<uchar>(i,j) = neg;
			}
		}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image",src);
		imshow("negative image",dst);
		waitKey();
	}
}

void testParcurgereSimplaDiblookStyle()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = src.clone();

		double t = (double)getTickCount(); // Get the current time [s]

		// the fastest approach using the “diblook style”
		uchar *lpSrc = src.data;
		uchar *lpDst = dst.data;
		int w = (int) src.step; // no dword alignment is done !!!
		for (int i = 0; i<height; i++)
			for (int j = 0; j < width; j++) {
				uchar val = lpSrc[i*w + j];
				lpDst[i*w + j] = 255 - val;
			}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image",src);
		imshow("negative image",dst);
		waitKey();
	}
}

void testColor2Gray()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src = imread(fname);

		int height = src.rows;
		int width = src.cols;

		Mat dst = Mat(height,width,CV_8UC1);

		// Asa se acceseaaza pixelii individuali pt. o imagine RGB 24 biti/pixel
		// Varianta ineficienta (lenta)
		for (int i=0; i<height; i++)
		{
			for (int j=0; j<width; j++)
			{
				Vec3b v3 = src.at<Vec3b>(i,j);
				uchar b = v3[0];
				uchar g = v3[1];
				uchar r = v3[2];
				dst.at<uchar>(i,j) = (r+g+b)/3;
			}
		}
		
		imshow("input image",src);
		imshow("gray image",dst);
		waitKey();
	}
}

void testBGR2HSV()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname);
		int height = src.rows;
		int width = src.cols;

		// Componentele d eculoare ale modelului HSV
		Mat H = Mat(height, width, CV_8UC1);
		Mat S = Mat(height, width, CV_8UC1);
		Mat V = Mat(height, width, CV_8UC1);

		// definire pointeri la matricele (8 biti/pixeli) folosite la afisarea componentelor individuale H,S,V
		uchar* lpH = H.data;
		uchar* lpS = S.data;
		uchar* lpV = V.data;

		Mat hsvImg;
		cvtColor(src, hsvImg, COLOR_BGR2HSV);

		// definire pointer la matricea (24 biti/pixeli) a imaginii HSV
		uchar* hsvDataPtr = hsvImg.data;

		for (int i = 0; i<height; i++)
		{
			for (int j = 0; j<width; j++)
			{
				int hi = i*width * 3 + j * 3;
				int gi = i*width + j;

				lpH[gi] = hsvDataPtr[hi] * 510 / 360;		// lpH = 0 .. 255
				lpS[gi] = hsvDataPtr[hi + 1];			// lpS = 0 .. 255
				lpV[gi] = hsvDataPtr[hi + 2];			// lpV = 0 .. 255
			}
		}

		imshow("input image", src);
		imshow("H", H);
		imshow("S", S);
		imshow("V", V);

		waitKey();
	}
}

void testResize()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		Mat dst1,dst2;
		//without interpolation
		resizeImg(src,dst1,320,false);
		//with interpolation
		resizeImg(src,dst2,320,true);
		imshow("input image",src);
		imshow("resized image (without interpolation)",dst1);
		imshow("resized image (with interpolation)",dst2);
		waitKey();
	}
}

void testCanny()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src,dst,gauss;
		src = imread(fname,IMREAD_GRAYSCALE);
		double k = 0.4;
		int pH = 50;
		int pL = (int) k*pH;
		GaussianBlur(src, gauss, Size(5, 5), 0.8, 0.8);
		Canny(gauss,dst,pL,pH,3);
		imshow("input image",src);
		imshow("canny",dst);
		waitKey();
	}
}

void testVideoSequence()
{
	VideoCapture cap("Videos/rubic.avi"); // off-line video from file
	//VideoCapture cap(0);	// live video from web cam
	if (!cap.isOpened()) {
		printf("Cannot open video capture device.\n");
		waitKey(0);
		return;
	}
		
	Mat edges;
	Mat frame;
	char c;

	while (cap.read(frame))
	{
		Mat grayFrame;
		cvtColor(frame, grayFrame, COLOR_BGR2GRAY);
		Canny(grayFrame,edges,40,100,3);
		imshow("source", frame);
		imshow("gray", grayFrame);
		imshow("edges", edges);
		c = waitKey(0);  // waits a key press to advance to the next frame
		if (c == 27) {
			// press ESC to exit
			printf("ESC pressed - capture finished\n"); 
			break;  //ESC pressed
		};
	}
}

bool isInside(Mat img, int i, int j) {
	if (i < 0 || i >= img.rows)
		return false;
	if (j < 0 || j >= img.cols)
		return false;
	return true;
}


void testSnap()
{
	VideoCapture cap(0); // open the deafult camera (i.e. the built in web cam)
	if (!cap.isOpened()) // openenig the video device failed
	{
		printf("Cannot open video capture device.\n");
		return;
	}

	Mat frame;
	char numberStr[256];
	char fileName[256];
	
	// video resolution
	Size capS = Size((int)cap.get(CAP_PROP_FRAME_WIDTH),
		(int)cap.get(CAP_PROP_FRAME_HEIGHT));

	// Display window
	const char* WIN_SRC = "Src"; //window for the source frame
	namedWindow(WIN_SRC, WINDOW_AUTOSIZE);
	moveWindow(WIN_SRC, 0, 0);

	const char* WIN_DST = "Snapped"; //window for showing the snapped frame
	namedWindow(WIN_DST, WINDOW_AUTOSIZE);
	moveWindow(WIN_DST, capS.width + 10, 0);

	char c;
	int frameNum = -1;
	int frameCount = 0;

	for (;;)
	{
		cap >> frame; // get a new frame from camera
		if (frame.empty())
		{
			printf("End of the video file\n");
			break;
		}

		++frameNum;
		
		imshow(WIN_SRC, frame);

		c = waitKey(10);  // waits a key press to advance to the next frame
		if (c == 27) {
			// press ESC to exit
			printf("ESC pressed - capture finished");
			break;  //ESC pressed
		}
		if (c == 115){ //'s' pressed - snapp the image to a file
			frameCount++;
			fileName[0] = NULL;
			sprintf(numberStr, "%d", frameCount);
			strcat(fileName, "Images/A");
			strcat(fileName, numberStr);
			strcat(fileName, ".bmp");
			bool bSuccess = imwrite(fileName, frame);
			if (!bSuccess) 
			{
				printf("Error writing the snapped image\n");
			}
			else
				imshow(WIN_DST, frame);
		}
	}

}


/* Histogram display function - display a histogram using bars (simlilar to L3 / PI)
Input:
name - destination (output) window name
hist - pointer to the vector containing the histogram values
hist_cols - no. of bins (elements) in the histogram = histogram image width
hist_height - height of the histogram image
Call example:
showHistogram ("MyHist", hist_dir, 255, 200);
*/
void showHistogram(const std::string& name, int* hist, const int  hist_cols, const int hist_height)
{
	Mat imgHist(hist_height, hist_cols, CV_8UC3, CV_RGB(255, 255, 255)); // constructs a white image

	//computes histogram maximum
	int max_hist = 0;
	for (int i = 0; i<hist_cols; i++)
	if (hist[i] > max_hist)
		max_hist = hist[i];
	double scale = 1.0;
	scale = (double)hist_height / max_hist;
	int baseline = hist_height - 1;

	for (int x = 0; x < hist_cols; x++) {
		Point p1 = Point(x, baseline);
		Point p2 = Point(x, baseline - cvRound(hist[x] * scale));
		line(imgHist, p1, p2, CV_RGB(255, 0, 255)); // histogram bins colored in magenta
	}

	imshow(name, imgHist);
}

void computeMultilevelThresholding() {
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		double t = (double)getTickCount(); // Get the current time [s]

		Mat src = imread(fname, IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		float hist[256];
		float pdf[256];
		for (int i = 0; i < 256; i++)
			hist[i] = 0;
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				hist[src.at<uchar>(i, j)]++;
			}
		}

		int max_hist = 0;

		for (int i = 0; i < 256; i++)
			if (hist[i] > max_hist)
				max_hist = hist[i];

		for (int i = 0; i < 256; i++)
			pdf[i] = (float)hist[i] / (float)max_hist;

		int wh = 5;
		float th = 0.0003;

		float ks[256];
		for (int i = 0; i < 256; i++)
			ks[i] = 0;

		std::vector<int> maxs;

		for (int k = wh; k < 255 - wh; k++) {
			float v = 0;
			bool ok = false;
			for (int x = -wh; x <= wh; x++) {
				v += pdf[k + x];
				if (pdf[k] < pdf[k + x])
					ok = true;
			}
			v = v / (2 * wh + 1);
			if (pdf[k] > (v + th) && ok == false) {
				ks[k] = pdf[k];
				maxs.push_back(k);
			}
		}

		maxs.push_back(0);
		maxs.push_back(255);
		std::sort(maxs.begin(),maxs.end());

		for (int i = 0; i < wh; i++)
			ks[i] = 0;
		for (int i = 255-wh; i < 256; i++)
			ks[i] =	1;

		int show[256];
		for (int i = 0; i < 256; i++)
			show[i] = ks[i] * max_hist;

		Mat dst = Mat(height, width, CV_8UC1);

		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				uchar val = src.at<uchar>(i, j);
				int minvalue = 100;
				int q = 0;
				for (int i = 0; i < maxs.size(); i++) {
					if (minvalue > std::abs(val - maxs[i])) {
						minvalue = std::abs(val - maxs[i]);
						q = maxs[i];
					}
				}
				dst.at<uchar>(i, j) = q;

			}
		}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);
		imshow("input image", src);
		imshow("dst", dst);
		showHistogram("norm", show, 256, 200);
		waitKey();
	}
}

void computeFloydSteinberg() {
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		double t = (double)getTickCount(); // Get the current time [s]

		Mat src = imread(fname, IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		float hist[256];
		float pdf[256];
		for (int i = 0; i < 256; i++)
			hist[i] = 0;
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				hist[src.at<uchar>(i, j)]++;
			}
		}

		int max_hist = 0;

		for (int i = 0; i < 256; i++)
			if (hist[i] > max_hist)
				max_hist = hist[i];

		for (int i = 0; i < 256; i++)
			pdf[i] = (float)hist[i] / (float)max_hist;

		int wh = 5;
		float th = 0.0003;

		float ks[256];
		for (int i = 0; i < 256; i++)
			ks[i] = 0;

		std::vector<int> maxs;

		for (int k = wh; k < 255 - wh; k++) {
			float v = 0;
			bool ok = false;
			for (int x = -wh; x <= wh; x++) {
				v += pdf[k + x];
				if (pdf[k] < pdf[k + x])
					ok = true;
			}
			v = v / (2 * wh + 1);
			if (pdf[k] > (v + th) && ok == false) {
				ks[k] = pdf[k];
				maxs.push_back(k);
			}
		}

		maxs.push_back(0);
		maxs.push_back(255);
		std::sort(maxs.begin(), maxs.end());

		for (int i = 0; i < wh; i++)
			ks[i] = 0;
		for (int i = 255 - wh; i < 256; i++)
			ks[i] = 1;

		int show[256];
		for (int i = 0; i < 256; i++)
			show[i] = ks[i] * max_hist;

		Mat dst = Mat(height, width, CV_8UC1);
		Mat dst2 = Mat(height, width, CV_8UC1);

		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				uchar val = src.at<uchar>(i, j);
				int minvalue = 100;
				int q = 0;
				for (int i = 0; i < maxs.size(); i++) {
					if (minvalue > std::abs(val - maxs[i])) {
						minvalue = std::abs(val - maxs[i]);
						q = maxs[i];
					}
				}
				dst.at<uchar>(i, j) = q;
				int error = val - q;
			}
		}

		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++) {
				int error = src.at<uchar>(i, j) - dst.at<uchar>(i, j);
				if (error > 0)
					int az = 0;
				if (isInside(src, i, j + 1))
					dst2.at<uchar>(i, j + 1) = max(0, min(dst.at<uchar>(i, j + 1) + (7 * error) / 16, 255));
				if (isInside(src, i + 1, j - 1))
					dst2.at<uchar>(i + 1, j - 1) = max(0, min(255, dst.at<uchar>(i + 1, j - 1) + (3 * error) / 16));
				if (isInside(src, i + 1, j))
					dst2.at<uchar>(i + 1, j) = max(0, min(255, dst.at<uchar>(i + 1, j) + (5 * error) / 16));
				if (isInside(src, i + 1, j + 1))
					dst2.at<uchar>(i + 1, j + 1) = max(0, min(255, dst.at<uchar>(i + 1, j + 1) + (error) / 16));
			}
		}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);
		imshow("input image", src);
		imshow("dst", dst);
		imshow("dst2", dst2);
		showHistogram("norm", show, 256, 200);
		waitKey();
	}
}

void computePDF() {
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		double t = (double)getTickCount(); // Get the current time [s]

		Mat src = imread(fname, IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		float hist[256];
		float pdf[256];
		for (int i = 0; i < 256; i++)
			hist[i] = 0;
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				hist[src.at<uchar>(i, j)]++;
			}
		}

		int max_hist = 0;

		for (int i = 0; i < 256; i++)
			if (hist[i] > max_hist)
				max_hist = hist[i];

		for (int i = 0; i < 256; i++)
			pdf[i] = (float)hist[i] / (float)max_hist;

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image", src);
		waitKey();
	}
}

void computeHistogram() {
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		double t = (double)getTickCount(); // Get the current time [s]

		Mat src = imread(fname, IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		int hist[255];
		for (int i = 0; i < 255; i++)
			hist[i] = 0;
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				hist[src.at<uchar>(i, j)]++;
			}
		}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image", src);
		showHistogram("histogram", hist, 255, 200);
		waitKey();
	}
}



void additivefactor(int x) {
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		double t = (double)getTickCount(); // Get the current time [s]

		Mat src = imread(fname, IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = Mat(height, width, CV_8UC1);
		// Asa se acceseaaza pixelii individuali pt. o imagine cu 8 biti/pixel
		// Varianta ineficienta (lenta)
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				uchar val = src.at<uchar>(i, j);
				uchar add = max(0,min(255,val + x));
				dst.at<uchar>(i, j) = add;
			}
		}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image", src);
		imshow("negative image", dst);
		waitKey();
	}
}

void multiplicativefactor(int x) {
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		double t = (double)getTickCount(); // Get the current time [s]

		Mat src = imread(fname, IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = Mat(height, width, CV_8UC1);
		// Asa se acceseaaza pixelii individuali pt. o imagine cu 8 biti/pixel
		// Varianta ineficienta (lenta)
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				uchar val = src.at<uchar>(i, j);
				uchar add = max(0, min(255, val * x));
				dst.at<uchar>(i, j) = add;
			}
		}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imwrite("D:\\Users\\Vicon\\IP\\Lab1\\OpenCVApplication-VS2019_OCV420_basic\\Images\\MultiplicativeImage.bmp", dst);

		imshow("input image", src);
		imshow("negative image", dst);


		waitKey();
	}
}


void createSquares() {
	Mat img(256, 256, CV_8UC3);
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			if (i < img.rows/2 && j < img.cols/2) {
				img.at<Vec3b>(i, j)[0] = 255;
				img.at<Vec3b>(i, j)[1] = 255;
				img.at<Vec3b>(i, j)[2] = 255;
			}
			else if (i < img.rows/2 && j>img.cols/2) {
				img.at<Vec3b>(i, j)[0] = 0;
				img.at<Vec3b>(i, j)[1] = 0;
				img.at<Vec3b>(i, j)[2] = 255;
			}
			else if (i > img.rows / 2 && j<img.cols / 2) {
				img.at<Vec3b>(i, j)[0] = 0;
				img.at<Vec3b>(i, j)[1] = 255;
				img.at<Vec3b>(i, j)[2] = 0;
			}
			else if (i > img.rows / 2 && j>img.cols / 2) {
				img.at<Vec3b>(i, j)[0] = 0;
				img.at<Vec3b>(i, j)[1] = 255;
				img.at<Vec3b>(i, j)[2] = 255;
			}

		}
	}
	imshow("src",img);
	waitKey();
}

void horizontalFlip() {
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		double t = (double)getTickCount(); // Get the current time [s]

		Mat src = imread(fname, IMREAD_COLOR);
		int height = src.rows;
		int width = src.cols;
		Mat dst = Mat(height, width, CV_8UC3);
		// Asa se acceseaaza pixelii individuali pt. o imagine cu 8 biti/pixel
		// Varianta ineficienta (lenta)
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				Vec3b val = src.at<Vec3b>(i, j);
				dst.at<Vec3b>(i, width - 1 - j) = val;
			}
		}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image", src);
		imshow("negative image", dst);


		waitKey();
	}
}

void verticalFlip() {
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		double t = (double)getTickCount(); // Get the current time [s]

		Mat src = imread(fname, IMREAD_COLOR);
		int height = src.rows;
		int width = src.cols;
		Mat dst = Mat(height, width, CV_8UC3);
		// Asa se acceseaaza pixelii individuali pt. o imagine cu 8 biti/pixel
		// Varianta ineficienta (lenta)
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				Vec3b val = src.at<Vec3b>(i, j);
				dst.at<Vec3b>(height - 1 - i, j) = val;
			}
		}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image", src);
		imshow("negative image", dst);


		waitKey();
	}
}

void centerCrop() {
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		double t = (double)getTickCount(); // Get the current time [s]

		Mat src = imread(fname, IMREAD_COLOR);
		int height = src.rows;
		int width = src.cols;
		Mat dst = Mat(height/2, width/2, CV_8UC3);
		// Asa se acceseaaza pixelii individuali pt. o imagine cu 8 biti/pixel
		// Varianta ineficienta (lenta)
		for (int i = height/4; i < 3*height/4; i++)
		{
			for (int j = width/4; j < 3*width/4; j++)
			{
				Vec3b val = src.at<Vec3b>(i, j);
				dst.at<Vec3b>(i-height/4, j-width/4) = val;
			}
		}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image", src);
		imshow("negative image", dst);


		waitKey();
	}
}

void resize(int n, int m) {
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		double t = (double)getTickCount(); // Get the current time [s]

		Mat src = imread(fname, IMREAD_COLOR);
		int height = src.rows;
		int width = src.cols;
		Mat dst = Mat(n, m, CV_8UC3);
		double ratioHeight = (double) height / (double) n;
		double ratioWidth = (double) width / (double) m;
		// Asa se acceseaaza pixelii individuali pt. o imagine cu 8 biti/pixel
		// Varianta ineficienta (lenta)
		for (int i = 0; i < n; i++)
		{
			for (int j = 0; j < m; j++)
			{
				double oldI = i * ratioHeight;
				double oldJ = j * ratioWidth;
				Vec3b val = src.at<Vec3b>(min(height-1,round(oldI)), min(width-1,round(oldJ)));
				dst.at<Vec3b>(i,j) = val;
			}
		}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image", src);
		imshow("negative image", dst);


		waitKey();
	}
}

void splitColors() {
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		double t = (double)getTickCount(); // Get the current time [s]

		Mat src = imread(fname, IMREAD_COLOR);
		int height = src.rows;
		int width = src.cols;
		Mat dstr = Mat(height, width, CV_8UC1);
		Mat dstg = Mat(height, width, CV_8UC1);
		Mat dstb = Mat(height, width, CV_8UC1);
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				Vec3b val = src.at<Vec3b>(i, j);
				dstr.at<uchar>(i, j) = val[2];
				dstg.at<uchar>(i, j) = val[1];
				dstb.at<uchar>(i, j) = val[0];
			}
		}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image", src);
		imshow("r", dstr);
		imshow("g", dstg);
		imshow("b", dstb);

		waitKey();
	}
}

void toGrayscale() {
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		double t = (double)getTickCount(); // Get the current time [s]

		Mat src = imread(fname, IMREAD_COLOR);
		int height = src.rows;
		int width = src.cols;
		Mat dst = Mat(height, width, CV_8UC1);
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				Vec3b val = src.at<Vec3b>(i, j);
				dst.at<uchar>(i, j) = (val[0] + val[1] + val[2]) / 3;
			}
		}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image", src);
		imshow("grayscale", dst);

		waitKey();
	}

}

Mat cvtBinary(Mat src) {
	int height = src.rows;
	int width = src.cols;
	Mat dst = Mat(height, width, CV_8UC1);
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			uchar val = src.at<uchar>(i, j);
			if (val < 128)
				dst.at<uchar>(i, j) = 0;
			else
				dst.at<uchar>(i, j) = 255;
		}
	}
	return dst;
}

void toBinary(int tr) {
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		double t = (double)getTickCount(); // Get the current time [s]

		Mat src = imread(fname, IMREAD_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = Mat(height, width, CV_8UC1);
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				uchar val = src.at<uchar>(i, j);
				if (val < tr)
					dst.at<uchar>(i, j) = 0;
				else
					dst.at<uchar>(i, j) = 255;
			}
		}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image", src);
		imshow("b&w", dst);

		waitKey();
	}

}

void toHSV() {
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		double t = (double)getTickCount(); // Get the current time [s]

		Mat src = imread(fname, IMREAD_COLOR);
		int height = src.rows;
		int width = src.cols;
		Mat dsth = Mat(height, width, CV_8UC1);
		Mat dsts = Mat(height, width, CV_8UC1);
		Mat dstv = Mat(height, width, CV_8UC1);
		Mat dsthsv = Mat(height, width, CV_8UC3);
		Mat dstConverted = Mat(height, width, CV_8UC3);
		for (int i = 0; i < height; i++)
		{
			for (int j = 0; j < width; j++)
			{
				Vec3b val = src.at<Vec3b>(i, j);
				double r = ((double)val[2]) / 255;
				double g = ((double)val[1]) / 255;
				double b = ((double)val[0]) / 255;
				double M = max(r, max(g, b));
				double m = min(r, min(g, b));
				double C = M - m;
				double h, s, v;
				v = M;
				if (M != 0) {
					s = C / M;
				}
				else {
					s = 0;
				}
				if (C != 0) {
					if (M == r) h = 60 * (g - b) / C;
					if (M == g) h = 120 + 60 * (b - r) / C;
					if (M == b) h = 240 + 60 * (r - g) / C;
				}
				else {
					h = 0;
				}
				if (h < 0)
					h += 360;

				dsth.at<uchar>(i, j) = h / 2;
				dsts.at<uchar>(i, j) = s * 255;
				dstv.at<uchar>(i, j) = v * 255;


				dsthsv.at<Vec3b>(i, j)[0] = dsth.at<uchar>(i, j);
				dsthsv.at<Vec3b>(i, j)[1] = dsts.at<uchar>(i, j);
				dsthsv.at<Vec3b>(i, j)[2] = dstv.at<uchar>(i, j);
			}
		}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		cvtColor(dsthsv, dstConverted, COLOR_HSV2BGR, 0);


		imshow("input image", src);
		imshow("h", dsth);
		imshow("s", dsts);
		imshow("v", dstv);

		imshow("converted", dstConverted);

		waitKey();
	}
}

void detectSign() {
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		double t = (double)getTickCount(); // Get the current time [s]

		Mat src = imread(fname, IMREAD_COLOR);
		int height = src.rows;
		int width = src.cols;
		Mat hsvImg = Mat(height, width, CV_8UC3);
		cv::cvtColor(src, hsvImg, COLOR_BGR2HSV);
		Mat hsv_channels[3];
		cv::split(hsvImg, hsv_channels);
		Mat mask;
		cv::inRange(hsv_channels[0], Scalar(10, 255, 255), Scalar(179, 255, 255), mask);
		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image", src);
		imshow("sign", mask);

		waitKey();
	}
}

int low_H = 0;
int high_H = 360;

Mat aux;

static void trackbar(int, void*)
{
	setTrackbarPos("Low H", "Slider", low_H);

	
	Mat hsvImg = Mat(aux.rows, aux.cols, CV_8UC3);
	cv::cvtColor(aux, hsvImg, COLOR_BGR2HSV);
	Mat hsv_channels[3];
	cv::split(hsvImg, hsv_channels);
	Mat mask;
	cv::inRange(hsv_channels[0], Scalar(low_H, 255, 255), Scalar(high_H, 255, 255), mask);

	imshow("sign", mask);

}

void trackbars() {
	char fname[MAX_PATH];

	while (openFileDlg(fname))
	{

		namedWindow("Slider");

		createTrackbar("Low H", "Slider", &low_H, 360, trackbar);
		createTrackbar("High H", "Slider", &high_H, 360, trackbar);

		Mat src = imread(fname, IMREAD_COLOR);
		int height = src.rows;
		int width = src.cols;
		aux = src;

		double t = (double)getTickCount(); // Get the current time [s]

		
		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image", src);
		

		waitKey();
	}
}

bool isObjectLabeled(Mat src, Vec3b label, int row, int col) {
	Vec3b value = src.at<Vec3b>(row, col);
	return value == label;
}

int computeArea(Mat src, Vec3b label) {
	int area = 0;
	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			if (isObjectLabeled(src, label, i, j))
				area++;
		}
	}
	return area;
}

int *computeCenterOfMass(Mat src, Vec3b label) {
	int area = computeArea(src, label);
	int sumr = 0;
	int sumc = 0;
	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			if (isObjectLabeled(src, label, i, j)) {
				sumr += i;
				sumc += j;
			}
		}
	}
	int *result = (int *)calloc(3,sizeof(int));
	result[0] = sumr / area;
	result[1] = sumc / area;
	return result;
}

int computeAxisOfElongation(Mat src, Vec3b label) {
	int nom = 0;
	int denom1 = 0, denom2 = 0;
	int* centerOfMass = (int*)calloc(3, sizeof(int));
	int* aux = computeCenterOfMass(src, label);
	centerOfMass[0] = aux[0];
	centerOfMass[1] = aux[1];
	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			if (isObjectLabeled(src, label, i, j)) {
				nom += (i - centerOfMass[0]) * (j - centerOfMass[1]);
				denom1 += (j - centerOfMass[1]) * (j - centerOfMass[1]);
				denom2 += (i - centerOfMass[0]) * (i - centerOfMass[0]);
			}
		}
	}
	if (nom == 0 && denom1 == denom2)
		return -1000;
	return (atan2(2 * nom, (denom1 - denom2))/2 + PI) * 180 / PI;
}

bool isContour(Mat src, Vec3b label, int i, int j) {
	if (isInside(src, i + 1, j + 1) && src.at<Vec3b>(i + 1, j + 1) != label)
		return true;
	if (isInside(src, i, j + 1) && src.at<Vec3b>(i, j + 1) != label)
		return true;
	if (isInside(src, i - 1, j + 1) && src.at<Vec3b>(i - 1, j + 1) != label)
		return true;
	if (isInside(src, i - 1, j) && src.at<Vec3b>(i - 1, j) != label)
		return true;
	if (isInside(src, i - 1, j - 1) && src.at<Vec3b>(i - 1, j - 1) != label)
		return true;
	if (isInside(src, i, j - 1) && src.at<Vec3b>(i, j - 1) != label)
		return true;
	if (isInside(src, i + 1, j - 1) && src.at<Vec3b>(i + 1, j - 1) != label)
		return true;
	if (isInside(src, i + 1, j) && src.at<Vec3b>(i + 1, j) != label)
		return true;
	return false;

}

int computePerimeter(Mat src, Vec3b label) {
	int perim = 0;
	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			if (isObjectLabeled(src, label, i, j)) {
				if (isContour(src, label, i, j))
					perim++;
			}
		}
	}
	return perim * PI/4;
}

float computeAspectRatio(Mat src, Vec3b label) {
	int maxr = -1, maxc = -1;
	int minr = src.rows, minc = src.cols;
	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			if (isObjectLabeled(src, label, i, j)) {
				if (j > maxc)
					maxc = j;
				if (j < minc)
					minc = j;
				if (i > maxr)
					maxr = i;
				if (i < minr)
					minr = i;

			}
		}
	}
	return (maxc - minc + 1.0f) / (maxr - minr + 1.0f);
}

int computeProjectionOnRow(Mat src, Vec3b label, int row) {
	int sum = 0;
	for (int i = 0; i < src.cols; i++)
		if (isObjectLabeled(src, label, row, i))
			sum++;
	return sum;
}

int computeProjectionOnCol(Mat src, Vec3b label, int col) {
	int sum = 0;
	for (int i = 0; i < src.rows; i++)
		if (isObjectLabeled(src, label, i, col))
			sum++;
	return sum;
}

void computeAttributesCallback(int event, int x, int y, int flags, void* param)
{
	Mat* src = (Mat*)param;
	//DOUBLE CLICK
	if (event == EVENT_LBUTTONDBLCLK)
	{
		printf("Pos(x,y): %d,%d  Color(RGB): %d,%d,%d\n",
			x, y,
			(int)(*src).at<Vec3b>(y, x)[2],
			(int)(*src).at<Vec3b>(y, x)[1],
			(int)(*src).at<Vec3b>(y, x)[0]);
		Vec3b label = (*src).at <Vec3b>(y, x);
		int* centerOfMass = (int*)calloc(3, sizeof(int));
		int* aux = computeCenterOfMass(*src, label);
		centerOfMass[0] = aux[0];
		centerOfMass[1] = aux[1];
		int area = computeArea(*src, label);
		int axis = computeAxisOfElongation(*src, label);
		int perim = computePerimeter(*src, label);
		float ratio = computeAspectRatio(*src, label);
		float thinness = 4.0f * PI * area / (perim * perim);
		printf("Area: %d\n", area);
		printf("Center of mass: row %d, col %d\n", centerOfMass[0], centerOfMass[1]);
		printf("Axis of elongation angle - %d degrees\n", axis);
		printf("Perimeter - %d\n", perim);
		printf("Thinness - %.2f\n", thinness);
		printf("Aspect ratio - %.2f\n", ratio);

		Mat dst = Mat(src->rows, src->cols, CV_8UC3);
		for (int i = 0; i < (*src).rows; i++) {
			for (int j = 0; j < (*src).cols; j++) {
				if (isObjectLabeled(*src, label, i, j)) {
					if (isContour(*src, label, i, j)) {
						dst.at<Vec3b>(i, j) = src->at<Vec3b>(i, j);
					}
				}
			}
		}

		circle(dst, Point(centerOfMass[1],centerOfMass[0]), 5, Scalar(255,0,255));

		Point p1 = Point(centerOfMass[1], centerOfMass[0]);

		//circle(dst, p1, 3, Scalar(0, 0, 255));

		Point p2, p3;

		p2.x = (int)round(p1.x + 50 * cos(axis * PI / 180.0));
		p2.y = (int)round(p1.y + 50 * sin(axis * PI / 180.0));

		p3.x = (int)round(p1.x - 50 * cos(axis * PI / 180.0));
		p3.y = (int)round(p1.y - 50 * sin(axis * PI / 180.0));

		line(dst, p2, p3, Scalar(label));

		Mat dst1 = Mat(src->rows, src->cols, CV_8UC3);

		Mat dst2 = Mat(src->rows, src->cols, CV_8UC3);

		for (int i = 0; i < src->rows; i++)
			for (int j = 0; j < computeProjectionOnRow(*src, label, i);j++)
				dst1.at<Vec3b>(i, j) = label;

		for (int i = 0; i < src->cols; i++)
			for (int j = 0; j < computeProjectionOnCol(*src, label, i); j++)
				dst2.at<Vec3b>(j, i) = label;

		imshow("destination", dst);
		imshow("destination1", dst1);
		imshow("destination2", dst2);
	}
}

void thresholdingCallback(Mat src, int thArea, int phiLow, int phiHigh)
{
	std::vector<Vec3b> labels;
	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			Vec3b label = src.at<Vec3b>(i, j);
			if (!std::count(labels.begin(), labels.end(), label))
				labels.push_back(label);
		}
	}

	std::vector<Vec3b> labelsAgain;

	Mat dst = Mat(src.rows, src.cols, CV_8UC3);
	for (int i = 0; i < labels.size(); i++) {
		int area = computeArea(src, labels[i]);
		int axis = computeAxisOfElongation(src, labels[i]);
		if (area < thArea&& phiLow < axis && axis < phiHigh)
			labelsAgain.push_back(labels[i]);
	}

	
	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			Vec3b label = src.at<Vec3b>(i, j);
			if (std::count(labelsAgain.begin(), labelsAgain.end(), label))
				dst.at<Vec3b>(i, j) = src.at<Vec3b>(i, j);
		}
	}
	imshow("LOL", dst);
}

void testMouseClick()
{
	Mat src;
	// Read image from file 
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		src = imread(fname);
		//Create a window
		namedWindow("My Window", 1);

		//set the callback function for any mouse event
		setMouseCallback("My Window", computeAttributesCallback, &src);

		//show the image
		imshow("My Window", src);

		// Wait until user press some key
		waitKey(0);
	}
}

std::vector<Vec2i> get4Neighbours(int i, int j) {
	int di[4] = { -1,0,1,0 };
	int dj[4] = { 0,-1,0,1 };
	std::vector<Vec2i> neighbours;
	for (int k = 0; k < 4; k++) {
		Vec2i vec;
		vec[0] = i + di[k];
		vec[1] = j + dj[k];
		neighbours.push_back(vec);
	}
	return neighbours;
}

std::vector<Vec2i> get8Neighbours(int i, int j) {
	int di[8] = { 0,-1,-1,-1,0,1,1,1 };
	int dj[8] = { 1,1,0,-1,-1,-1,0,1 };
	std::vector<Vec2i> neighbours;
	for (int k = 0; k < 8; k++) {
		Vec2i vec;
		vec[0] = i + di[k];
		vec[1] = j + dj[k];
		neighbours.push_back(vec);
	}
	return neighbours;
}

std::vector<Vec2i> getPNeighbours(Mat src, int i, int j) {
	int di[4] = { -1,-1,-1,0 };
	int dj[4] = { -1,0,1,-1 };
	std::vector<Vec2i> neighbours;
	for (int k = 0; k < 4; k++) {
		Vec2i vec;
		vec[0] = i + di[k];
		vec[1] = j + dj[k];
		if(isInside(src,vec[0],vec[1]))
			neighbours.push_back(vec);
	}
	return neighbours;
}

void bfsLabel() {
	Mat src;

	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		src = imread(fname, IMREAD_GRAYSCALE);

		src = cvtBinary(src);

		int height = src.rows;
		int width = src.cols;

		Mat dst = Mat(height, width, CV_8UC3);

		int label = 0;
		Mat labels = Mat(height, width, CV_32SC1);

		for (int i = 0; i < height; i++)
			for (int j = 0; j < width; j++)
				labels.at<int>(i, j) = 0;

		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				uchar val = src.at<uchar>(i, j);
				if (val == 0 && labels.at<int>(i,j) == 0) {
					label++;
					std::queue<Point2i> Q;
					labels.at<int>(i,j) = label;
					Q.push({ i,j });
					while (!Q.empty()) {
						Point2i q = Q.front();
						Q.pop();
						std::vector<Vec2i> neighbours = get8Neighbours(q.x, q.y);
						for (int n = 0; n < neighbours.size(); n++) {
							if(isInside(src, neighbours.at(n)[0], neighbours.at(n)[1]))
								if(labels.at<int>(neighbours.at(n)[0], neighbours.at(n)[1]) == 0)
									if (src.at<uchar>(neighbours.at(n)[0], neighbours.at(n)[1]) == 0) {
										labels.at<int>(neighbours.at(n)[0], neighbours.at(n)[1]) = label;
										Q.push({ neighbours.at(n)[0],neighbours.at(n)[1] });
									}
						}
					}
				}
			}
		}


		Mat labelColors = Mat(label + 1, 1, CV_8UC3);
		for (int i = 0; i < label; i++) {
			labelColors.at<Vec3b>(i, 0)[0] = 0;
			labelColors.at<Vec3b>(i, 0)[1] = 0;
			labelColors.at<Vec3b>(i, 0)[2] = 0;
		}
		std::default_random_engine gen;
		std::uniform_int_distribution<int> d(0, 255);
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				//printf("%d\n", labels.at<int>(i, j));
				if (labelColors.at<Vec3b>(labels.at<int>(i,j),0)[0] == 0 && labelColors.at<Vec3b>(labels.at<int>(i, j), 0)[1] == 0 && labelColors.at<Vec3b>(labels.at<int>(i, j), 0)[2] == 0) {
					uchar b = d(gen);
					uchar g = d(gen);
					uchar r = d(gen);
					labelColors.at<Vec3b>(labels.at<int>(i, j), 0)[0] = b;
					labelColors.at<Vec3b>(labels.at<int>(i, j), 0)[1] = g;
					labelColors.at<Vec3b>(labels.at<int>(i, j), 0)[2] = r;
				}
				dst.at<Vec3b>(i, j)[0] = labelColors.at<Vec3b>(labels.at<int>(i, j), 0)[0];
				dst.at<Vec3b>(i, j)[1] = labelColors.at<Vec3b>(labels.at<int>(i, j), 0)[1];
				dst.at<Vec3b>(i, j)[2] = labelColors.at<Vec3b>(labels.at<int>(i, j), 0)[2];
			}
		}

		imshow("source", src);
		imshow("labeled", dst);
		waitKey(0);
	}


}



void twoPassLabel() {
	Mat src;

	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		src = imread(fname, IMREAD_GRAYSCALE);

		src = cvtBinary(src);

		int height = src.rows;
		int width = src.cols;

		Mat dstint = Mat(height, width, CV_8UC3);
		Mat dst = Mat(height, width, CV_8UC3);

		int label = 0;
		Mat labels = Mat(height, width, CV_32SC1);

		for (int i = 0; i < height; i++)
			for (int j = 0; j < width; j++)
				labels.at<int>(i, j) = 0;

		std::vector<std::vector<int>> edges;

		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				if (src.at<uchar>(i, j) == 0 && labels.at<int>(i, j) == 0) {
					std::vector<int> L;
					for each (Vec2i neighbour in getPNeighbours(src, i, j))
					{
						if (labels.at<int>(neighbour[0], neighbour[1]) > 0)
							L.push_back(labels.at<int>(neighbour[0], neighbour[1]));
					}
					if (L.size() == 0) {
						label++;
						edges.resize(label + 1);
						labels.at<int>(i, j) = label;
					}
					else {
						int x = *std::min_element(L.begin(), L.end());
						labels.at<int>(i, j) = x;
						for (int y = 0; y < L.size(); y++) {
							if (L.at(y) != x) {
								edges.at(x).push_back(L.at(y));
								edges.at(L.at(y)).push_back(x);
							}
						}
					}
				}
			}
		}

		Mat labelColors = Mat(label + 1, 1, CV_8UC3);
		for (int i = 0; i < label; i++) {
			labelColors.at<Vec3b>(i, 0)[0] = 0;
			labelColors.at<Vec3b>(i, 0)[1] = 0;
			labelColors.at<Vec3b>(i, 0)[2] = 0;
		}
		std::default_random_engine gen;
		std::uniform_int_distribution<int> d(0, 255);
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				//printf("%d\n", labels.at<int>(i, j));
				if (labelColors.at<Vec3b>(labels.at<int>(i, j), 0)[0] == 0 && labelColors.at<Vec3b>(labels.at<int>(i, j), 0)[1] == 0 && labelColors.at<Vec3b>(labels.at<int>(i, j), 0)[2] == 0) {
					uchar b = d(gen);
					uchar g = d(gen);
					uchar r = d(gen);
					labelColors.at<Vec3b>(labels.at<int>(i, j), 0)[0] = b;
					labelColors.at<Vec3b>(labels.at<int>(i, j), 0)[1] = g;
					labelColors.at<Vec3b>(labels.at<int>(i, j), 0)[2] = r;
				}
				dstint.at<Vec3b>(i, j)[0] = labelColors.at<Vec3b>(labels.at<int>(i, j), 0)[0];
				dstint.at<Vec3b>(i, j)[1] = labelColors.at<Vec3b>(labels.at<int>(i, j), 0)[1];
				dstint.at<Vec3b>(i, j)[2] = labelColors.at<Vec3b>(labels.at<int>(i, j), 0)[2];
			}
		}


		int newlabel = 0;
		Mat newlabels = Mat(label + 1, 1, CV_32SC1);

		for (int i = 0; i < label + 1; i++)
				newlabels.at<int>(i, 0) = 0;

		for (int i = 1; i < label + 1; i++) {
			if (newlabels.at<int>(i, 0) == 0) {
				newlabel++;
				std::queue<int> Q;
				newlabels.at<int>(i, 0) = newlabel;
				Q.push(i);
				while (!Q.empty()) {
					int x = Q.front();
					Q.pop();
					for (int y = 0; y < edges.at(x).size(); y++) {
						if (newlabels.at<int>(edges.at(x).at(y), 0) == 0) {
							newlabels.at<int>(edges.at(x).at(y), 0) = newlabel;
							Q.push(edges.at(x).at(y));
						}
					}
				}
			}
		}

		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				labels.at<int>(i, j) = newlabels.at<int>(labels.at<int>(i, j), 0);
			}
		}

		for (int i = 0; i < label; i++) {
			labelColors.at<Vec3b>(i, 0)[0] = 0;
			labelColors.at<Vec3b>(i, 0)[1] = 0;
			labelColors.at<Vec3b>(i, 0)[2] = 0;
		}
		for (int i = 0; i < height; i++) {
			for (int j = 0; j < width; j++) {
				//printf("%d\n", labels.at<int>(i, j));
				if (labelColors.at<Vec3b>(labels.at<int>(i, j), 0)[0] == 0 && labelColors.at<Vec3b>(labels.at<int>(i, j), 0)[1] == 0 && labelColors.at<Vec3b>(labels.at<int>(i, j), 0)[2] == 0) {
					uchar b = d(gen);
					uchar g = d(gen);
					uchar r = d(gen);
					labelColors.at<Vec3b>(labels.at<int>(i, j), 0)[0] = b;
					labelColors.at<Vec3b>(labels.at<int>(i, j), 0)[1] = g;
					labelColors.at<Vec3b>(labels.at<int>(i, j), 0)[2] = r;
				}
				dst.at<Vec3b>(i, j)[0] = labelColors.at<Vec3b>(labels.at<int>(i, j), 0)[0];
				dst.at<Vec3b>(i, j)[1] = labelColors.at<Vec3b>(labels.at<int>(i, j), 0)[1];
				dst.at<Vec3b>(i, j)[2] = labelColors.at<Vec3b>(labels.at<int>(i, j), 0)[2];
			}
		}

		imshow("intermediate", dstint);
		imshow("destination", dst);
		imshow("source", src);
		waitKey(0);

	}
}

void contour(Mat image) {
	Mat dst = Mat(image.rows, image.cols, CV_8UC3);
	std::vector<Point2i> border;
	Point2i startPoint;
	bool found = false;
	Vec3b color = Vec3b(255,255,255);
	for (int i = 0; i < image.rows && !found; i++) {
		for (int j = 0; j < image.cols && !found; j++) {
			if (color != image.at<Vec3b>(i, j)) {
				found = true;
				color = image.at<Vec3b>(i, j);
				startPoint = Point2i(i, j);
				border.push_back(startPoint);
			}
		}
	}
	if (!found)
		return;
	int dir = 7;
	Point2i currentPoint;
	currentPoint = startPoint;
	do {
		std::vector<Vec2i> neighbours = get8Neighbours(currentPoint.x, currentPoint.y);
		found = false;
		int currDir;
		if (dir % 2 == 0)
			currDir = (dir + 7) % 8;
		else
			currDir = (dir + 6) % 8;
		while (!found) {
			if (isInside(image, neighbours.at(currDir)[0], neighbours.at(currDir)[1]) && image.at<Vec3b>(neighbours.at(currDir)[0], neighbours.at(currDir)[1]) != Vec3b(255,255,255)) {
				found = true;
				dir = currDir;
				currentPoint = Point2i(neighbours.at(currDir)[0], neighbours.at(currDir)[1]);
				border.push_back(currentPoint);
			}
			currDir = (currDir + 1) % 8;
		}

	} while (border.size() < 3 || border.at(1) != border.at(border.size() - 1) || border.at(border.size()-2) != border.at(0));

	border.pop_back();

	border.pop_back();

	for each (Point2i point in border) {
		dst.at<Vec3b>(point.x, point.y) = color;
	}

	imshow("border", dst);
	imshow("original", image);
	waitKey(0);
}

void chain(Mat image) {
	Mat dst = Mat(image.rows, image.cols, CV_8UC3);
	std::vector<Point2i> border;
	std::vector<int> chain;
	std::vector<int> derivative;
	Point2i startPoint;
	bool found = false;
	Vec3b color = Vec3b(255, 255, 255);
	for (int i = 0; i < image.rows && !found; i++) {
		for (int j = 0; j < image.cols && !found; j++) {
			if (color != image.at<Vec3b>(i, j)) {
				found = true;
				color = image.at<Vec3b>(i, j);
				startPoint = Point2i(i, j);
				border.push_back(startPoint);
			}
		}
	}
	if (!found)
		return;
	int dir = 7;
	Point2i currentPoint;
	currentPoint = startPoint;
	do {
		std::vector<Vec2i> neighbours = get8Neighbours(currentPoint.x, currentPoint.y);
		found = false;
		int currDir;
		if (dir % 2 == 0)
			currDir = (dir + 7) % 8;
		else
			currDir = (dir + 6) % 8;
		while (!found) {
			if (isInside(image, neighbours.at(currDir)[0], neighbours.at(currDir)[1]) && image.at<Vec3b>(neighbours.at(currDir)[0], neighbours.at(currDir)[1]) != Vec3b(255, 255, 255)) {
				found = true;
				int der = currDir - dir;
				if (der < 0)
					der += 8;
				dir = currDir;
				currentPoint = Point2i(neighbours.at(currDir)[0], neighbours.at(currDir)[1]);
				border.push_back(currentPoint);
				if (border.at(0) != border.at(border.size() - 1) && border.at(border.size() - 2) != border.at(0)) {
					chain.push_back(dir);
				}
				derivative.push_back(der);
			}
			currDir = (currDir + 1) % 8;
		}

	} while (border.size() < 3 || border.at(1) != border.at(border.size() - 1) || border.at(border.size() - 2) != border.at(0));

	border.pop_back();

	border.pop_back();

	for each (Point2i point in border) {
		dst.at<Vec3b>(point.x, point.y) = color;
	}

	printf("\nChain code:\n");
	for each (int x in chain) {
		printf("%d ", x);
	}

	printf("\nDerivative code:\n");
	for each (int x in derivative) {
		printf("%d ", x);
	}

	imshow("original", image);
	waitKey(0);
}


void detectBorder() {
	Mat src;

	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		src = imread(fname);
		contour(src);
	}
}

void chainCode() {
	Mat src;

	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		src = imread(fname);
		chain(src);
	}
}

void draw(int i, int j, std::vector<int> chain) {
	int di[8] = { 0,-1,-1,-1,0,1,1,1 };
	int dj[8] = { 1,1,0,-1,-1,-1,0,1 };
	Point2i curr = Point2i(i, j);
	Mat dst = imread(".\\Images\\gray_background.bmp", IMREAD_COLOR);
	dst.at<Vec3b>(i, j) = Vec3b(0, 255, 255);
	int dir = 0;
	for each (int change in chain) {
		dir = change;
		curr = Point2i(curr.x + di[dir], curr.y + dj[dir]);
		dst.at<Vec3b>(curr.x, curr.y) = Vec3b(0, 255, 255);
	}

	imshow("dst", dst);
	waitKey(0);
}

void drawChain() {
	FILE* fp;
	char name[50];

	fp = fopen(".\\Images\\reconstruct.txt", "r");

	if (fp == NULL)
	{
		printf("Error opening file\n");
		exit(1);
	}

	int i, j;

	fscanf(fp, "%d %d\n", &i, &j);

	int nb;

	fscanf(fp, "%d\n", &nb);

	int chain;
	std::vector<int> chainCode;

	for (int i = 0; i < nb; i++) {
		fscanf(fp, "%d ", &chain);
		chainCode.push_back(chain);
	}

	fclose(fp);

	draw(i, j, chainCode);
}

Mat dilation(Mat image, Mat kernel) {
	Mat dst = Mat(image.rows, image.cols, CV_8UC1);
	for (int i = 0; i < image.rows; i++) 
		for (int j = 0; j < image.cols; j++) 
			dst.at<uchar>(i, j) = 255;
	for (int i = 0; i < image.rows; i++) {
		for (int j = 0; j < image.cols; j++) {
			if (image.at<uchar>(i, j) == 0) {
				for (int k = 0; k < kernel.rows; k++) {
					for (int l = 0; l < kernel.cols; l++) {
						int row = i - kernel.rows / 2 + k;
						int col = j - kernel.cols / 2 + l;
						if (isInside(image, row, col) && kernel.at<uchar>(k, l) == 0) {
							dst.at<uchar>(row, col) = 0;
						}
					}
				}
			}
		}
	}
	return dst;
}

void dilateImage() {
	Mat src;

	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		src = imread(fname, IMREAD_GRAYSCALE);
		src = cvtBinary(src);
		Mat kernel = Mat(3, 3, CV_8UC1);
		kernel.at<uchar>(0, 0) = 255;
		kernel.at<uchar>(2, 0) = 255;
		kernel.at<uchar>(0, 2) = 255;
		kernel.at<uchar>(2, 2) = 255;
		kernel.at<uchar>(1, 1) = 0;
		kernel.at<uchar>(0, 1) = 0;
		kernel.at<uchar>(1, 0) = 0;
		kernel.at<uchar>(1, 2) = 0;
		kernel.at<uchar>(2, 1) = 0;
		Mat dst = dilation(src, kernel);
		imshow("src",src);
		imshow("dst", dst);
		waitKey(0);

	}
}

void dilateImageNTimes() {
	int n;
	scanf("%d", &n);
	Mat src;

	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		src = imread(fname, IMREAD_GRAYSCALE);
		src = cvtBinary(src);
		Mat kernel = Mat(3, 3, CV_8UC1);
		kernel.at<uchar>(0, 0) = 255;
		kernel.at<uchar>(2, 0) = 255;
		kernel.at<uchar>(0, 2) = 255;
		kernel.at<uchar>(2, 2) = 255;
		kernel.at<uchar>(1, 1) = 0;
		kernel.at<uchar>(0, 1) = 0;
		kernel.at<uchar>(1, 0) = 0;
		kernel.at<uchar>(1, 2) = 0;
		kernel.at<uchar>(2, 1) = 0;
		Mat dst = dilation(src, kernel);
		Mat dstN = dilation(src, kernel);
		for (int i = 0; i < n - 2; i++) {
			dstN = dilation(dstN, kernel);
		}
		imshow("src", src);
		imshow("dst", dst);
		imshow("dstN", dstN);
		waitKey(0);

	}
}

Mat erosion(Mat image, Mat kernel) {
	Mat dst = Mat(image.rows, image.cols, CV_8UC1);
	for (int i = 0; i < image.rows; i++)
		for (int j = 0; j < image.cols; j++)
			dst.at<uchar>(i, j) = 255;
	for (int i = 0; i < image.rows; i++) {
		for (int j = 0; j < image.cols; j++) {
			if (image.at<uchar>(i, j) == 0) {
				bool ok = true;
				for (int k = 0; k < kernel.rows; k++) {
					for (int l = 0; l < kernel.cols; l++) {
						int row = i - kernel.rows / 2 + k;
						int col = j - kernel.cols / 2 + l;
						if (isInside(image, row, col) && kernel.at<uchar>(k, l) == 0 && image.at<uchar>(row, col) != 0) {
							ok = false;
						}
					}
				}
				if (ok) {
					dst.at<uchar>(i, j) = 0;
				}
			}
		}
	}
	return dst;
}

void erodeImage() {
	Mat src;

	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		src = imread(fname, IMREAD_GRAYSCALE);
		src = cvtBinary(src);
		Mat kernel = Mat(3, 3, CV_8UC1);
		kernel.at<uchar>(0, 0) = 255;
		kernel.at<uchar>(2, 0) = 255;
		kernel.at<uchar>(0, 2) = 255;
		kernel.at<uchar>(2, 2) = 255;
		kernel.at<uchar>(1, 1) = 0;
		kernel.at<uchar>(0, 1) = 0;
		kernel.at<uchar>(1, 0) = 0;
		kernel.at<uchar>(1, 2) = 0;
		kernel.at<uchar>(2, 1) = 0;
		Mat dst = erosion(src, kernel);
		imshow("src", src);
		imshow("dst", dst);
		waitKey(0);

	}
}

void erodeImageNTimes() {
	int n;
	scanf("%d", &n);
	Mat src;

	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		src = imread(fname, IMREAD_GRAYSCALE);
		src = cvtBinary(src);
		Mat kernel = Mat(3, 3, CV_8UC1);
		kernel.at<uchar>(0, 0) = 255;
		kernel.at<uchar>(2, 0) = 255;
		kernel.at<uchar>(0, 2) = 255;
		kernel.at<uchar>(2, 2) = 255;
		kernel.at<uchar>(1, 1) = 0;
		kernel.at<uchar>(0, 1) = 0;
		kernel.at<uchar>(1, 0) = 0;
		kernel.at<uchar>(1, 2) = 0;
		kernel.at<uchar>(2, 1) = 0;
		Mat dst = erosion(src, kernel);
		Mat dstN = erosion(src, kernel);
		for (int i = 0; i < n - 2; i++) {
			dstN = erosion(dstN, kernel);
		}
		imshow("src", src);
		imshow("dst", dst);
		imshow("dstN", dstN);
		waitKey(0);

	}
}

Mat open(Mat image, Mat kernel) {
	Mat dst = dilation(image, kernel);
	dst = erosion(dst, kernel);
	return dst;
}

void openImage() {
	Mat src;

	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		src = imread(fname, IMREAD_GRAYSCALE);
		src = cvtBinary(src);
		Mat kernel = Mat(3, 3, CV_8UC1);
		kernel.at<uchar>(0, 0) = 255;
		kernel.at<uchar>(2, 0) = 255;
		kernel.at<uchar>(0, 2) = 255;
		kernel.at<uchar>(2, 2) = 255;
		kernel.at<uchar>(1, 1) = 0;
		kernel.at<uchar>(0, 1) = 0;
		kernel.at<uchar>(1, 0) = 0;
		kernel.at<uchar>(1, 2) = 0;
		kernel.at<uchar>(2, 1) = 0;
		Mat dst = open(src, kernel);
		imshow("src", src);
		imshow("dst", dst);
		waitKey(0);

	}
}

void openImageNTimes() {
	int n;
	scanf("%d", &n);
	Mat src;

	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		src = imread(fname, IMREAD_GRAYSCALE);
		src = cvtBinary(src);
		Mat kernel = Mat(3, 3, CV_8UC1);
		kernel.at<uchar>(0, 0) = 255;
		kernel.at<uchar>(2, 0) = 255;
		kernel.at<uchar>(0, 2) = 255;
		kernel.at<uchar>(2, 2) = 255;
		kernel.at<uchar>(1, 1) = 0;
		kernel.at<uchar>(0, 1) = 0;
		kernel.at<uchar>(1, 0) = 0;
		kernel.at<uchar>(1, 2) = 0;
		kernel.at<uchar>(2, 1) = 0;
		Mat dst = open(src, kernel);
		Mat dstN = open(src, kernel);
		for (int i = 0; i < n - 2; i++) {
			dstN = open(dstN, kernel);
		}
		imshow("src", src);
		imshow("dst", dst);
		imshow("dstN", dstN);
		waitKey(0);

	}
}

Mat close(Mat image, Mat kernel) {
	Mat dst = erosion(image, kernel);
	dst = dilation(dst, kernel);
	return dst;
}

void closeImage() {
	Mat src;

	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		src = imread(fname, IMREAD_GRAYSCALE);
		src = cvtBinary(src);
		Mat kernel = Mat(3, 3, CV_8UC1);
		kernel.at<uchar>(0, 0) = 255;
		kernel.at<uchar>(2, 0) = 255;
		kernel.at<uchar>(0, 2) = 255;
		kernel.at<uchar>(2, 2) = 255;
		kernel.at<uchar>(1, 1) = 0;
		kernel.at<uchar>(0, 1) = 0;
		kernel.at<uchar>(1, 0) = 0;
		kernel.at<uchar>(1, 2) = 0;
		kernel.at<uchar>(2, 1) = 0;
		Mat dst = close(src, kernel);
		imshow("src", src);
		imshow("dst", dst);
		waitKey(0);

	}
}

void closeImageNTimes() {
	int n;
	scanf("%d", &n);
	Mat src;

	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		src = imread(fname, IMREAD_GRAYSCALE);
		src = cvtBinary(src);
		Mat kernel = Mat(3, 3, CV_8UC1);
		kernel.at<uchar>(0, 0) = 255;
		kernel.at<uchar>(2, 0) = 255;
		kernel.at<uchar>(0, 2) = 255;
		kernel.at<uchar>(2, 2) = 255;
		kernel.at<uchar>(1, 1) = 0;
		kernel.at<uchar>(0, 1) = 0;
		kernel.at<uchar>(1, 0) = 0;
		kernel.at<uchar>(1, 2) = 0;
		kernel.at<uchar>(2, 1) = 0;
		Mat dst = close(src, kernel);
		Mat dstN = close(src, kernel);
		for (int i = 0; i < n - 2; i++) {
			dstN = close(dstN, kernel);
		}
		imshow("src", src);
		imshow("dst", dst);
		imshow("dstN", dstN);
		waitKey(0);

	}
}

Mat subtract(Mat A, Mat B) {
	Mat dst = Mat(A.rows, B.cols, CV_8UC1);
	for (int i = 0; i < A.rows; i++)
		for (int j = 0; j < A.cols; j++)
			dst.at<uchar>(i, j) = 255;
	for (int i = 0; i < A.rows; i++) {
		for (int j = 0; j < A.cols; j++) {
			if (A.at<uchar>(i, j) == 0)
				if (B.at<uchar>(i, j) == 255) 
					dst.at<uchar>(i, j) = 0;
		}
	}
	return dst;
}

Mat boundaryExtract(Mat image) {
	Mat kernel = Mat(3, 3, CV_8UC1);
	for (int i = 0; i < 3; i++)
		for (int j = 0; j < 3; j++)
			kernel.at<uchar>(i, j) = 0;
	Mat er = erosion(image, kernel);
	//imshow("src", er);
	//waitKey(1);
	Mat dst = subtract(image, er);
	return dst;
}

void boundaryExtractImage() {
	Mat src;

	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		src = imread(fname, IMREAD_GRAYSCALE);
		src = cvtBinary(src);
		Mat dst = boundaryExtract(src);
		imshow("src", src);
		imshow("dst", dst);
		waitKey(0);

	}
}

Mat complement(Mat image) {
	Mat dst = Mat(image.rows, image.cols, CV_8UC1);
	for (int i = 0; i < image.rows; i++) {
		for (int j = 0; j < image.cols; j++) {
			if (image.at<uchar>(i, j) == 0) {
				dst.at<uchar>(i, j) = 255;
			}
			else {
				dst.at<uchar>(i, j) = 0;
			}
		}
	}
	return dst;
}

Mat intersection(Mat A, Mat B) {
	Mat dst = Mat(A.rows, B.cols, CV_8UC1);
	for (int i = 0; i < A.rows; i++) {
		for (int j = 0; j < A.cols; j++) {
			if (A.at<uchar>(i, j) == 0 && B.at<uchar>(i, j) == 0)
				dst.at<uchar>(i, j) = 0;
			else
				dst.at<uchar>(i, j) = 255;
		}
	}
	return dst;
}

Mat imageUnion(Mat A, Mat B) {
	Mat dst = Mat(A.rows, B.cols, CV_8UC1);
	for (int i = 0; i < A.rows; i++) {
		for (int j = 0; j < A.cols; j++) {
			if (A.at<uchar>(i, j) == 0 || B.at<uchar>(i, j) == 0)
				dst.at<uchar>(i, j) = 0;
			else
				dst.at<uchar>(i, j) = 255;
		}
	}
	return dst;
}

bool checkEquality(Mat A, Mat B) {
	for (int i = 0; i < A.rows; i++) {
		for (int j = 0; j < A.cols; j++) {
			if (A.at<uchar>(i, j) != B.at<uchar>(i, j))
				return false;
		}
	}
	return true;
}

Mat fillRegion(Mat image) {
	Mat dst = Mat(image.rows, image.cols, CV_8UC1);
	for (int i = 0; i < image.rows; i++)
		for (int j = 0; j < image.cols; j++)
			dst.at<uchar>(i, j) = 255;

	dst.at<uchar>(image.rows / 2, image.cols / 2) = 0;

	imshow("dst", dst);
	waitKey(0);
	Mat kernel = Mat(3, 3, CV_8UC1);
	kernel.at<uchar>(0, 0) = 255;
	kernel.at<uchar>(2, 0) = 255;
	kernel.at<uchar>(0, 2) = 255;
	kernel.at<uchar>(2, 2) = 255;
	kernel.at<uchar>(1, 1) = 0;
	kernel.at<uchar>(0, 1) = 0;
	kernel.at<uchar>(1, 0) = 0;
	kernel.at<uchar>(1, 2) = 0;
	kernel.at<uchar>(2, 1) = 0;
	Mat prev = dst;
	Mat imageC = complement(image);
	dst = intersection(dilation(dst, kernel), imageC);
	while (!checkEquality(prev, dst)) {
		prev = dst;
		dst = intersection(dilation(dst, kernel), complement(image));
		imshow("dst", dst);
		waitKey(1);
	}
	return imageUnion(image, dst);
}

void fillRegionImage() {
	Mat src;

	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		src = imread(fname, IMREAD_GRAYSCALE);
		src = cvtBinary(src);
		Mat dst = fillRegion(src);
		imshow("src", src);
		imshow("dst", dst);
		waitKey(0);

	}
}

float meanIntensityValue(Mat src) {
	float sum = 0;
	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			sum += src.at<uchar>(i, j);
		}
	}
	float mean = sum / (src.rows * src.cols);
	return mean;
}

float standardDeviation(Mat src) {
	float sum = 0;
	float squareSum = 0;
	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			uchar entry = src.at<uchar>(i, j);
			sum += entry;
			squareSum += entry * entry;
		}
	}
	float mean = sum / (src.rows * src.cols);
	double variance = squareSum / (src.rows * src.cols) - mean * mean;
	return sqrt(variance);
}

int *computeHistogramVector(Mat src) {
	int* arr = (int*)calloc(256, sizeof(int));
	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			uchar entry = src.at<uchar>(i, j);
			arr[entry]++;
		}
	}
	return arr;
}

int* computeCumultavieHistogram(Mat src) {
	int* arr = computeHistogramVector(src);
	int* arrC = (int*)calloc(256, sizeof(int));
	arrC[0] = arr[0];
	for (int i = 1; i < 256; i++) {
		arrC[i] = arrC[i - 1] + arr[i];
	}
	return arrC;
}

void computeHistograms() {
	Mat src;

	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		int* arr = computeHistogramVector(src);
		int* arrC = (int *)calloc(256, sizeof(int));
		int max = -1;
		int maxC = -1;
		arrC[0] = arr[0];
		for (int i = 0; i < 256; i++) {
			if (max < arr[i]) {
				max = arr[i];
			}
			if (i > 0) {
				arrC[i] = arr[i] + arrC[i - 1];
				if (max < arrC[i]) {
					maxC = arrC[i];
				}
			}
		}
		showHistogram("simple", arr, 255, 200);
		showHistogram("cumulative", arrC, 255, 200);
		printf("Mean: %f\nDeviation: %f\n", meanIntensityValue(src), standardDeviation(src));
	}
}

int findThreshold(Mat src, int error) {
	int* arr = computeHistogramVector(src);
	int imin = -1;
	int imax = -1;
	for (int i = 1; i < 256; i++) {
		if (arr[i] > 0) {
			imax = i;
		}
		if (arr[i] > 0 && imin == -1) {
			imin = i;
		}
	}
	int Tprev = (imax + imin)/2;
	int T = Tprev + error + 1;
	while (std::abs(T - Tprev) > error) {
		int n1 = 0;
		int n2 = 0;
		float mean1 = 0;
		float mean2 = 0;
		for (int i = imin; i <= T; i++) {
			n1 += arr[i];
			mean1 += i * arr[i];
		}
		mean1 = mean1 / n1;
		for (int i = T + 1; i <= imax; i++) {
			n2 += arr[i];
			mean2 += i * arr[i];
		}
		mean2 = mean2 / n2;
		Tprev = T;
		T = (mean1 + mean2) / 2;
	}
	return T;
}

Mat cvtBinaryWithT(Mat src, int T) {
	int height = src.rows;
	int width = src.cols;
	Mat dst = Mat(height, width, CV_8UC1);
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			uchar val = src.at<uchar>(i, j);
			if (val < T)
				dst.at<uchar>(i, j) = 0;
			else
				dst.at<uchar>(i, j) = 255;
		}
	}
	return dst;
}

void automaticThresholding() {
	Mat src;

	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		int t = findThreshold(src, 2);
		printf("Threshold: %d\n", t);
		Mat dst = cvtBinaryWithT(src, t);
		imshow("src", src);
		imshow("dst", dst);
		waitKey(0);
	}
}

Mat strechShrink(Mat src, int minout, int maxout) {
	int* arr = computeHistogramVector(src);
	int imin = -1;
	int imax = -1;
	for (int i = 1; i < 256; i++) {
		if (arr[i] > 0) {
			imax = i;
		}
		if (arr[i] > 0 && imin == -1) {
			imin = i;
		}
	}
	int height = src.rows;
	int width = src.cols;
	Mat dst = Mat(height, width, CV_8UC1);
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			uchar val = src.at<uchar>(i, j);
			dst.at<uchar>(i, j) = minout + (val - imin) * (maxout - minout) / (imax - imin);
		}
	}
	return dst;
}

void strechShrinkImage() {
	Mat src;

	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		int min, max;
		printf("Input min limit: ");
		scanf("%d", &min);
		printf("Input max limit: ");
		scanf("%d", &max);
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		Mat dst = strechShrink(src, min, max);
		int* arrS = computeHistogramVector(src);
		int* arrD = computeHistogramVector(dst);
		showHistogram("arrS", arrS, 256, 200);
		showHistogram("arrD", arrD, 256, 200);
		imshow("src", src);
		imshow("dst", dst);
		waitKey(0);
	}
}

Mat gammaCorrection(Mat src, float gamma) {
	int height = src.rows;
	int width = src.cols;
	Mat dst = Mat(height, width, CV_8UC1);
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			uchar val = src.at<uchar>(i, j);
			dst.at<uchar>(i, j) = max(0, min(255, pow((float(val) / 255.0), gamma)*255));
		}
	}
	return dst;
}

void gammaCorrectImage() {
	Mat src;

	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		float gamma;
		printf("Input gamma: ");
		scanf("%f",&gamma);
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		Mat dst = gammaCorrection(src, gamma);
		int* arrS = computeHistogramVector(src);
		int* arrD = computeHistogramVector(dst);
		showHistogram("arrS", arrS, 256, 200);
		showHistogram("arrD", arrD, 256, 200);
		imshow("src", src);
		imshow("dst", dst);
		waitKey(0);
	}
}

Mat slide(Mat src, int offset) {
	int height = src.rows;
	int width = src.cols;
	Mat dst = Mat(height, width, CV_8UC1);
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			uchar val = src.at<uchar>(i, j);
			dst.at<uchar>(i, j) = max(0, min(255, val + offset));
		}
	}
	return dst;
}

void slideImage() {
	Mat src;

	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		int offset;
		printf("Input offset: ");
		scanf("%d", &offset);
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		Mat dst = slide(src, offset);
		int* arrS = computeHistogramVector(src);
		int* arrD = computeHistogramVector(dst);
		showHistogram("arrS", arrS, 256, 200);
		showHistogram("arrD", arrD, 256, 200);
		imshow("src", src);
		imshow("dst", dst);
		waitKey(0);
	}
}

Mat eqHistogram(Mat src) {
	int height = src.rows;
	int width = src.cols;
	Mat dst = Mat(height, width, CV_8UC1);
	int* histc = computeCumultavieHistogram(src);
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			uchar val = src.at<uchar>(i, j);
			dst.at<uchar>(i, j) = (float)255 / (src.cols * src.rows) * histc[src.at<uchar>(i, j)];
		}
	}
	return dst;
}

void eqImage() {
	Mat src;

	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		Mat dst = eqHistogram(src);
		int* arrS = computeHistogramVector(src);
		int* arrD = computeHistogramVector(dst);
		showHistogram("arrS", arrS, 256, 200);
		showHistogram("arrD", arrD, 256, 200);
		imshow("src", src);
		imshow("dst", dst);
		waitKey(0);
	}
}

Mat convolution(Mat filter, Mat img) {

	Mat output = Mat(img.rows, img.cols, CV_8UC1);

	int scalingCoeff = 1;
	int additionFactor = 0;

	//TODO: decide if the filter is low pass or high pass and compute the scaling coefficient and the addition factor
	// low pass if all elements >= 0
	// high pass has elements < 0

	bool low = true;
	int sum = 0;
	int sumPlus = 0;
	int sumMinus = 0;

	for (int i = 0; i < filter.rows; i++) {
		for (int j = 0; j < filter.cols; j++) {
			int value = filter.at<int>(i, j);
			if (value < 0) {
				low = false;
				sumMinus += -value;
			}
			else {
				sumPlus += value;
			}
			sum += value;
		}
	}

	// compute scaling coefficient and addition factor for low pass and high pass
	// low pass: additionFactor = 0, scalingCoeff = sum of all elements
	// high pass: formula 9.20


	if (low) {
		scalingCoeff = sum;
	}
	else {
		scalingCoeff = 2 * (sumPlus > sumMinus ? sumPlus : sumMinus);
		additionFactor = 255 / 2;
	}


	// TODO: implement convolution operation (formula 9.2)
	// do not forget to divide with the scaling factor and add the addition factor in order to have values between [0, 255]

	int k1 = filter.rows / 2;
	int k2 = filter.cols / 2;
	for (int i = k1; i < img.rows - k1; i++) {
		for (int j = k2; j < img.cols - k2; j++) {
			int S = 0;
			for (int u = 0; u < filter.rows; u++) {
				for (int v = 0; v < filter.cols; v++) {
					S += filter.at<int>(u, v) * img.at<uchar>(i + u - k1, j + v - k2);
				}
			}
			output.at<uchar>(i, j) = S / scalingCoeff + additionFactor;
		}
	}

	return output;
}

void applyFilters() {

	Mat src;

	// LOW PASS
	// mean filter 5x5
	int meanFilterData5x5[25] = { 1, 1, 1, 1, 1, 1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1 };
	Mat meanFilter5x5 = Mat(5, 5, CV_32SC1);
	for (int i = 0; i < 5; i++)
		for (int j = 0; j < 5; j++)
			meanFilter5x5.at<int>(i,j) = meanFilterData5x5[i * 5 + j];

	// mean filter 3x3
	Mat meanFilter3x3 = Mat(3, 3, CV_32SC1);
	for (int i = 0; i < 3; i++)
		for (int j = 0; j < 3; j++)
			meanFilter3x3.at<int>(i,j) = meanFilterData5x5[i * 3 + j];

	// gaussian filter
	int gaussianFilterData[9] = { 1, 2, 1, 2, 4, 2, 1, 2, 1 };
	Mat gaussianFilter = Mat(3, 3, CV_32SC1);
	for (int i = 0; i < 3; i++)
		for (int j = 0; j < 3; j++)
			gaussianFilter.at<int>(i,j) = gaussianFilterData[i * 3 + j];

	// HIGH PASS
	// laplace filter 3x3
	int laplaceFilterData[9] = { -1, -1, -1, -1, 8, -1, -1, -1, -1 };
	Mat laplaceFilter = Mat(3, 3, CV_32SC1);
	for (int i = 0; i < 3; i++)
		for (int j = 0; j < 3; j++)
			laplaceFilter.at<int>(i,j) = laplaceFilterData[i * 3 + j];

	int highpassFilterData[9] = { -1, -1, -1, -1, 9, -1, -1, -1, -1 };
	Mat highPassFilter = Mat(3, 3, CV_32SC1);
	for (int i = 0; i < 3; i++)
		for (int j = 0; j < 3; j++)
			highPassFilter.at<int>(i,j) = highpassFilterData[i * 3 + j];

	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		Mat output = convolution(meanFilter3x3, src);
		imshow("output", output);
		waitKey(0);
		output = convolution(meanFilter5x5, src);
		imshow("output", output);
		waitKey(0);
		output = convolution(gaussianFilter, src);
		imshow("output", output);
		waitKey(0);
		output = convolution(laplaceFilter, src);
		imshow("output", output);
		waitKey(0);
		output = convolution(highPassFilter, src);
		imshow("output", output);
		waitKey(0);
	}

	
}

void centering_transform(Mat img) {
	//expects floating point image
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			img.at<float>(i, j) = ((i + j) & 1) ? -img.at<float>(i, j) : img.at<float>(i, j);
		}
	}
}

Mat generic_frequency_domain_filter(Mat src, int type, int param)
{
	int height = src.rows;
	int width = src.cols;

	Mat srcf;
	src.convertTo(srcf, CV_32FC1);
	// Centering transformation
	centering_transform(srcf);

	//perform forward transform with complex image output
	Mat fourier;
	dft(srcf, fourier, DFT_COMPLEX_OUTPUT);

	//split into real and imaginary channels fourier(i, j) = Re(i, j) + i * Im(i, j)
	Mat channels[] = { Mat::zeros(src.size(), CV_32F), Mat::zeros(src.size(), CV_32F) };
	split(fourier, channels);  // channels[0] = Re (real part), channels[1] = Im (imaginary part)

	//calculate magnitude and phase in floating point images mag and phi
	// http://www3.ncc.edu/faculty/ens/schoenf/elt115/complex.html
	// from cartesian to polar coordinates

	Mat mag, phi;
	magnitude(channels[0], channels[1], mag);
	phase(channels[0], channels[1], phi);


	// TODO: Display here the log of magnitude (Add 1 to the magnitude to avoid log(0)) (see image 9.4e))
	// do not forget to normalize
	Mat logMag = Mat(height, width, CV_32FC1);
	for (int i = 0; i < height; i++)
		for (int j = 0; j < width; j++)
			logMag.at<float>(i,j) = log(mag.at<float>(i, j) + 1);

	Mat logMat;
	normalize(logMag, logMat, 0, 255, NORM_MINMAX, CV_8UC1);
	imshow("log", logMat);

	// TODO: Insert filtering operations here ( channels[0] = Re(DFT(I), channels[1] = Im(DFT(I) )
	for (int u = 0; u < height; u++)
		for (int v = 0; v < height; v++)
			if (type == 0) {
				if (pow((height / 2 - u), 2) + pow((width / 2 - v), 2) > param) {
					channels[0].at<float>(u, v) = 0;
					channels[1].at<float>(u, v) = 0;
				}
			}
			else if (type == 1) {
				if (pow((height / 2 - u), 2) + pow((width / 2 - v), 2) <= param) {
					channels[0].at<float>(u, v) = 0;
					channels[1].at<float>(u, v) = 0;
				}
			}
			else if (type == 2) {
				channels[0].at<float>(u, v) = channels[0].at<float>(u, v) * exp(-(pow((height / 2 - u), 2) + pow((width / 2 - v), 2))/param);
				channels[1].at<float>(u, v) = channels[1].at<float>(u, v) * exp(-(pow((height / 2 - u), 2) + pow((width / 2 - v), 2)) / param);
			}
			else if (type == 3) {
				channels[0].at<float>(u, v) = channels[0].at<float>(u, v) * (1 - exp(-(pow((height / 2 - u), 2) + pow((width / 2 - v), 2)) / param));
				channels[1].at<float>(u, v) = channels[1].at<float>(u, v) * (1 - exp(-(pow((height / 2 - u), 2) + pow((width / 2 - v), 2)) / param));
			}


	//perform inverse transform and put results in dstf
	Mat dst, dstf;
	merge(channels, 2, fourier);
	dft(fourier, dstf, DFT_INVERSE | DFT_REAL_OUTPUT);

	// Inverse Centering transformation
	centering_transform(dstf);

	//normalize the result and put in the destination image
	normalize(dstf, dst, 0, 255, NORM_MINMAX, CV_8UC1);

	return dst;
}

void applyFreqFilter() {
	Mat src;
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		Mat dst1 = generic_frequency_domain_filter(src, 0, 20);
		Mat dst2 = generic_frequency_domain_filter(src, 1, 20);
		Mat dst3 = generic_frequency_domain_filter(src, 2, 100);
		Mat dst4 = generic_frequency_domain_filter(src, 3, 100);
		imshow("src", src);
		imshow("dst1", dst1);
		imshow("dst2", dst2);
		imshow("dst3", dst3);
		imshow("dst4", dst4);
		waitKey(0);
	}
}

Mat medianFilter(Mat src, int w)
{
	Mat dst = Mat(src.rows, src.cols, CV_8UC1);
	for (int i = 0; i < src.rows; i++) {
		for(int j=0;j<src.cols;j++){
			std::vector<uchar> intensities;
			for (int u = 0; u < w; u++) {
				for (int v = 0; v < w; v++) {
					int index = i + u - w / 2;
					int jndex = j + v - w / 2;
					if (isInside(src, index, jndex)) {
						intensities.push_back(src.at<uchar>(index, jndex));
					}
				}
			}
			sort(intensities.begin(), intensities.end());
			dst.at<uchar>(i, j) = intensities.at(intensities.size() / 2);
		}
	}
	return dst;
}

void applyMedianFilter() {
	Mat src;
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		int n;
		scanf("%d", &n);
		Mat dst = medianFilter(src, n);
		imshow("src", src);
		imshow("dst", dst);
		waitKey(0);
	}
}

Mat buildGaussianFilter(double sigma, int w, int aux) {
	Mat filter = Mat(w, w, CV_32SC1);
	for (int i = 0; i < w; i++) {
		for (int j = 0; j < w; j++) {
			int exponent = (i - w / 2) * (i - w / 2) + (j - w / 2) * (j - w / 2);
			exponent = exponent / (2 * sigma * sigma);
			exponent = -exponent;
			filter.at<int>(i, j) = aux * (exp(exponent) / (2 * PI * sigma * sigma));
		}
	}
	for (int i = 0; i < w; i++) {
		for (int j = 0; j < w; j++) {
			printf("%d ", filter.at<int>(i, j));
		}
		printf("\n");
	}
	return filter;
}

Mat obtainMiddleX(Mat filter) {
	Mat middle = Mat(1, filter.cols, CV_32SC1);
	for (int i = 0; i < filter.cols; i++) {
		middle.at<int>(0, i) = filter.at<int>(filter.rows / 2, i);
	}
	return middle;
}

Mat obtainMiddleXWithSigma(int w, double sigma, int aux) {
	Mat middle = Mat(1, w, CV_32SC1);
	for (int i = 0; i < w; i++) {
		int exponent = (i - w / 2) * (i - w / 2);
		exponent = exponent / (2 * sigma * sigma);
		exponent = -exponent;
		middle.at<int>(0, i) = aux * (exp(exponent) / sqrt(2 * PI * sigma * sigma));
	}
	return middle;
}

Mat obtainMiddleY(Mat filter) {
	Mat middle = Mat(filter.rows, 1, CV_32SC1);
	for (int i = 0; i < filter.rows; i++) {
		middle.at<int>(i, 0) = filter.at<int>(i, filter.cols / 2);
	}
	return middle;
}

Mat obtainMiddleYWithSigma(int w, double sigma, int aux) {
	Mat middle = Mat(w, 1, CV_32SC1);
	for (int i = 0; i < w; i++) {
		int exponent = (i - w / 2) * (i - w / 2);
		exponent = exponent / (2 * sigma * sigma);
		exponent = -exponent;
		middle.at<int>(i, 0) = aux * (exp(exponent) / sqrt(2 * PI * sigma * sigma));
	}
	return middle;
}


void applyGaussianFilter() 
{
	Mat src;
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		int w, aux;
		double sigma;
		printf("\nW: ");
		scanf("%d", &w);
		printf("\nSigma: ");
		scanf("%lf", &sigma);
		printf("\nMult: ");
		scanf("%d", &aux);
		Mat dst = convolution(buildGaussianFilter(sigma, w, aux), src);
		imshow("src", src);
		imshow("dst", dst);
		waitKey(0);
	}
}

void applyVectorGaussian()
{
	Mat src;
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, IMREAD_GRAYSCALE);
		int w, aux;
		double sigma;
		printf("\nW: ");
		scanf("%d", &w);
		printf("\nSigma: ");
		scanf("%lf", &sigma);
		printf("\nMult: ");
		scanf("%d", &aux);
		Mat dst1 = convolution(buildGaussianFilter(sigma, w, aux), src);
		Mat filterX = obtainMiddleXWithSigma(w, sigma, aux);
		Mat filterY = obtainMiddleYWithSigma(w, sigma, aux);
		Mat dst = convolution(filterX, convolution(filterY, src));
		imshow("src", src);
		imshow("dst", dst);
		imshow("dst1", dst1);
		waitKey(0);
	}
}

int main()
{
	int op;
	int tr;
	do
	{
		system("cls");
		destroyAllWindows();
		printf("Menu:\n");
		printf(" 1 - Median Filter\n");
		printf(" 2 - Gaussian Filter\n");
		printf(" 3 - Gaussian Vector Filter\n");
		printf(" 0 - Exit\n\n");
		printf("Option: ");
		scanf("%d",&op);
		switch (op)
		{
		case 1:
			applyMedianFilter();
			break;
		case 2:
			applyGaussianFilter();
			break;
		case 3:
			applyVectorGaussian();
			break;
		}
	}
	while (op!=0);
	return 0;
}