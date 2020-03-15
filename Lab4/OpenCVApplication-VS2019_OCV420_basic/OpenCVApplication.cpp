// OpenCVApplication.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "common.h"


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
	int result[2];
	result[0] = sumr / area;
	result[1] = sumc / area;
	return result;
}

int computeAxisOfElongation(Mat src, Vec3b label) {
	int nom = 0;
	int denom1 = 0, denom2 = 0;
	int* centerOfMass = (int*)calloc(2, sizeof(int));
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
	return (atan2(2 * nom, denom1 - denom2)/2) * 180 / PI;
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
	if (event == EVENT_LBUTTONDOWN)
	{
		printf("Pos(x,y): %d,%d  Color(RGB): %d,%d,%d\n",
			x, y,
			(int)(*src).at<Vec3b>(y, x)[2],
			(int)(*src).at<Vec3b>(y, x)[1],
			(int)(*src).at<Vec3b>(y, x)[0]);
		Vec3b label = (*src).at <Vec3b>(y, x);
		int* centerOfMass = (int*)calloc(2, sizeof(int));
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

		dst.at<Vec3b>(centerOfMass[0], centerOfMass[1]) = src->at<Vec3b>(centerOfMass[0], centerOfMass[1]);

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

Mat thresholdingCallback(Mat src, int thArea, int phiLow, int phiHigh)
{
	std::vector<Vec3b> labels;
	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			Vec3b label = src.at<Vec3b>(i, j);
			if (!std::count(labels.begin(), labels.end(), label))
				labels.push_back(label);
		}
	}

	for (int i = 0; i < labels.size(); i++) {
		int area = computeArea(src, labels[i]);
		int axis = computeAxisOfElongation(src, labels[i]);
		if (area > thArea)
			labels.erase(labels.begin() + i - 1);
		if(axis < phiLow || axis > phiHigh)
			labels.erase(labels.begin() + i - 1);
	}

	Mat dst = Mat(src.rows, src.cols, CV_8UC3);
	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			Vec3b label = src.at<Vec3b>(i, j);
			if (std::count(labels.begin(), labels.end(), label))
				dst.at<Vec3b>(i, j) = src.at<Vec3b>(i, j);
		}
	}
	return dst;
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

int main()
{
	int op;
	int tr;
	do
	{
		system("cls");
		destroyAllWindows();
		printf("Menu:\n");
		printf(" 1 - Test Mouse Click\n");
		printf(" 0 - Exit\n\n");
		printf("Option: ");
		scanf("%d",&op);
		switch (op)
		{
			case 1:
				testMouseClick();
				break;
		}
	}
	while (op!=0);
	return 0;
}