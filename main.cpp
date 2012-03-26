#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgproc/imgproc_c.h"

#include "Homography.h"

#include <iostream>

using namespace cv;

int main(int argc, char* argv[])
{
  cvNamedWindow("Img1",CV_WINDOW_AUTOSIZE);
  cvNamedWindow("Img2",CV_WINDOW_AUTOSIZE);
  cvNamedWindow("Result",CV_WINDOW_AUTOSIZE);

	IplImage *img = cvLoadImage("input1.png");
	IplImage *img2 = cvLoadImage("input2.png");

	CvMat *homography= cvCreateMat(3,3,CV_32F);
	Homography(img,img2,homography);
  cvShowImage("Img1",img);
  cvShowImage("Img2",img2);

	cvWarpPerspective(img, img2, homography, CV_INTER_NN+CV_WARP_FILL_OUTLIERS, cvScalar(0));
  cvShowImage("Result",img2);
	cvWaitKey();

	cvDestroyWindow("Img1");
	cvDestroyWindow("Img2");
	cvDestroyWindow("Result");

	cvReleaseMat(&homography);
	cvReleaseImage(&img);
	cvReleaseImage(&img2);

	return 0;
}
