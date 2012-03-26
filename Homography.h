#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgproc/imgproc_c.h"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/ml/ml.hpp"
#include "opencv2/calib3d/calib3d.hpp"

#include <vector>

using namespace cv;

#define USE_FLANN

double compareSURFDescriptors( const float* d1, const float* d2, double best, int length );

int naiveNearestNeighbor( const float* vec, int laplacian,
	const CvSeq* model_keypoints,
	const CvSeq* model_descriptors );

void findPairs( const CvSeq* objectKeypoints, const CvSeq* objectDescriptors,
	const CvSeq* imageKeypoints, const CvSeq* imageDescriptors, std::vector<int>& ptpairs );

void flannFindPairs( const CvSeq*, const CvSeq* objectDescriptors,
	const CvSeq*, const CvSeq* imageDescriptors, std::vector<int>& ptpairs );

void drawSurfResult(IplImage* img, CvSeq* seq, CvScalar color);

void Homography(IplImage* frame1, IplImage* frame2, CvMat* homography);
