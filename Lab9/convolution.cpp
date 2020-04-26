#include "stdafx.h"
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;


void convolution(Mat_<int> &filter, Mat_<uchar> &img, Mat_<uchar> &output) {

    output.create(img.size());
    memcpy(output.data, img.data, img.rows * img.cols * sizeof(uchar));

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

    int k = filter.rows / 2;
    for (int i = k; i < img.rows - k; i++) {
        for (int j = k; j < img.cols - k; j++) {
            int S = 0;
            for (int u = 0; u < filter.rows; u++) {
                for (int v = 0; v < filter.cols; v++) {
                    S += filter.at<int>(u, v) * img.at<int>(i + u - k, j + v - k);
                }
            }
            output.at<int>(i, j) = S / scalingCoeff + additionFactor;
        }
    }
}


/*  in the frequency domain, the process of convolution simplifies to multiplication => faster than in the spatial domain
    the output is simply given by F(u,v)×G(u,v) where F(u,v) and G(u,v) are the Fourier transforms of their respective functions
    The frequency-domain representation of a signal carries information about the signal's magnitude and phase at each frequency*/

/*
The algorithm for filtering in the frequency domain is:
    a) Perform the image centering transform on the original image (9.15)
    b) Perform the DFT transform
    c) Alter the Fourier coefficients according to the required filtering
    d) Perform the IDFT transform
    e) Perform the image centering transform again (this undoes the first centering transform)
 */

void centering_transform(Mat img){
//expects floating point image
    for (int i = 0; i < img.rows; i++){
        for (int j = 0; j < img.cols; j++){
            img.at<float>(i, j) = ((i + j) & 1) ? -img.at<float>(i, j) : img.at<float>(i, j);
        }
    }
}

Mat generic_frequency_domain_filter(Mat src)
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

    // TODO: Insert filtering operations here ( channels[0] = Re(DFT(I), channels[1] = Im(DFT(I) )


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

int main() {


    // PART 1: convolution in the spatial domain
    Mat_<uchar> img = imread("cameraman.bmp", IMREAD_GRAYSCALE);
    Mat_<uchar> outputImage;

    // LOW PASS
    // mean filter 5x5
    int meanFilterData5x5[25];
    fill_n(meanFilterData5x5, 25, 1);
    Mat_<int> meanFilter5x5(5, 5, meanFilterData5x5);

    // mean filter 3x3
    Mat_<int> meanFilter3x3(3, 3, meanFilterData5x5);

    // gaussian filter
    int gaussianFilterData[9] = {1, 2, 1, 2, 4, 2, 1, 2, 1};
    Mat_<int> gaussianFilter(3, 3, gaussianFilterData);

    // HIGH PASS
    // laplace filter 3x3
    int laplaceFilterData[9] = {-1, -1, -1, -1, 8, -1, -1, -1, -1};
    Mat_<int> laplaceFilter(3, 3, laplaceFilterData);

    int highpassFilterData[9] = {-1, -1, -1, -1, 9, -1, -1, -1, -1};
    Mat_<int> highpassFilter(3, 3, highpassFilterData);

    //TODO: convolution with the mean filter 5 x 5
    //TODO: convolution with the mean filter 3 x 3
    //TODO: convolution with the gaussian filter
    //TODO: convolution with the laplacian filter
    //TODO: convolution with the highpass filter


    // PART 2: convolution in the frequency domain
    // use the generic_frequency_domain_filter() function

    // TODO: convolution with the ideal low pass filter (formula 9.16) take R^2 = 20
    // TODO: convolution with the ideal high pass filter (formula 9.17) take R^2 = 20
    // TODO: convolution with the Gaussian low pass filter (formula 9.18) take A = 10
    // TODO: convolution with the Gaussian high pass filter (formula 9.19) take A = 10

    return 0;
}