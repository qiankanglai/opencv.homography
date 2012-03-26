#include "Homography.h"

// opencv\samples\c\find_obj.cpp (Homography)
double compareSURFDescriptors( const float* d1, const float* d2, double best, int length )
{
	double total_cost = 0;
	assert( length % 4 == 0 );
	for( int i = 0; i < length; i += 4 )
	{
		double t0 = d1[i] - d2[i];
		double t1 = d1[i+1] - d2[i+1];
		double t2 = d1[i+2] - d2[i+2];
		double t3 = d1[i+3] - d2[i+3];
		total_cost += t0*t0 + t1*t1 + t2*t2 + t3*t3;
		if( total_cost > best )
			break;
	}
	return total_cost;
}

int naiveNearestNeighbor( const float* vec, int laplacian,
	const CvSeq* model_keypoints,
	const CvSeq* model_descriptors )
{
	int length = (int)(model_descriptors->elem_size/sizeof(float));
	int i, neighbor = -1;
	double d, dist1 = 1e6, dist2 = 1e6;
	CvSeqReader reader, kreader;
	cvStartReadSeq( model_keypoints, &kreader, 0 );
	cvStartReadSeq( model_descriptors, &reader, 0 );

	for( i = 0; i < model_descriptors->total; i++ )
	{
		const CvSURFPoint* kp = (const CvSURFPoint*)kreader.ptr;
		const float* mvec = (const float*)reader.ptr;
		CV_NEXT_SEQ_ELEM( kreader.seq->elem_size, kreader );
		CV_NEXT_SEQ_ELEM( reader.seq->elem_size, reader );
		if( laplacian != kp->laplacian )
			continue;
		d = compareSURFDescriptors( vec, mvec, dist2, length );
		if( d < dist1 )
		{
			dist2 = dist1;
			dist1 = d;
			neighbor = i;
		}
		else if ( d < dist2 )
			dist2 = d;
	}
	if ( dist1 < 0.6*dist2 )
		return neighbor;
	return -1;
}

void findPairs( const CvSeq* objectKeypoints, const CvSeq* objectDescriptors,
	const CvSeq* imageKeypoints, const CvSeq* imageDescriptors, std::vector<int>& ptpairs )
{
	int i;
	CvSeqReader reader, kreader;
	cvStartReadSeq( objectKeypoints, &kreader );
	cvStartReadSeq( objectDescriptors, &reader );
	ptpairs.clear();

	for( i = 0; i < objectDescriptors->total; i++ )
	{
		const CvSURFPoint* kp = (const CvSURFPoint*)kreader.ptr;
		const float* descriptor = (const float*)reader.ptr;
		CV_NEXT_SEQ_ELEM( kreader.seq->elem_size, kreader );
		CV_NEXT_SEQ_ELEM( reader.seq->elem_size, reader );
		int nearest_neighbor = naiveNearestNeighbor( descriptor, kp->laplacian, imageKeypoints, imageDescriptors );
		if( nearest_neighbor >= 0 )
		{
			ptpairs.push_back(i);
			ptpairs.push_back(nearest_neighbor);
		}
	}
}

void flannFindPairs( const CvSeq*, const CvSeq* objectDescriptors,
	const CvSeq*, const CvSeq* imageDescriptors, std::vector<int>& ptpairs )
{
	int length = (int)(objectDescriptors->elem_size/sizeof(float));

	cv::Mat m_object(objectDescriptors->total, length, CV_32F);
	cv::Mat m_image(imageDescriptors->total, length, CV_32F);

	// copy descriptors
	CvSeqReader obj_reader;
	float* obj_ptr = m_object.ptr<float>(0);
	cvStartReadSeq( objectDescriptors, &obj_reader );
	for(int i = 0; i < objectDescriptors->total; i++ )
	{
		const float* descriptor = (const float*)obj_reader.ptr;
		CV_NEXT_SEQ_ELEM( obj_reader.seq->elem_size, obj_reader );
		memcpy(obj_ptr, descriptor, length*sizeof(float));
		obj_ptr += length;
	}
	CvSeqReader img_reader;
	float* img_ptr = m_image.ptr<float>(0);
	cvStartReadSeq( imageDescriptors, &img_reader );
	for(int i = 0; i < imageDescriptors->total; i++ )
	{
		const float* descriptor = (const float*)img_reader.ptr;
		CV_NEXT_SEQ_ELEM( img_reader.seq->elem_size, img_reader );
		memcpy(img_ptr, descriptor, length*sizeof(float));
		img_ptr += length;
	}

	// find nearest neighbors using FLANN
	cv::Mat m_indices(objectDescriptors->total, 2, CV_32S);
	cv::Mat m_dists(objectDescriptors->total, 2, CV_32F);
	cv::flann::Index flann_index(m_image, cv::flann::KDTreeIndexParams(4));  // using 4 randomized kdtrees
	flann_index.knnSearch(m_object, m_indices, m_dists, 2, cv::flann::SearchParams(64) ); // maximum number of leafs checked

	int* indices_ptr = m_indices.ptr<int>(0);
	float* dists_ptr = m_dists.ptr<float>(0);
	for (int i=0;i<m_indices.rows;++i) {
		if (dists_ptr[2*i]<0.6*dists_ptr[2*i+1]) {
			ptpairs.push_back(i);
			ptpairs.push_back(indices_ptr[2*i]);
		}
	}
}

void drawSurfResult(IplImage* img, CvSeq* seq, CvScalar color)
{
	for(int i = 0; i < seq->total; i++ )
	{
		CvSURFPoint* r = (CvSURFPoint*)cvGetSeqElem( seq, i );
		CvPoint center;
		int radius;
		center.x = cvRound(r->pt.x);
		center.y = cvRound(r->pt.y);
		radius = cvRound(r->size*1.2/9.*2);
		cvCircle( img, center, radius, color);
	}
}

void Homography(IplImage* frame1, IplImage* frame2, CvMat* homography)
{
	//Extract SURF points by initializing parameters
	//SURF is better than SIFT
	CvMemStorage* storage = cvCreateMemStorage(0);
	IplImage* grayimage = cvCreateImage(cvGetSize(frame1), 8, 1);
	CvSeq *kp1=NULL, *kp2=NULL; 
	CvSeq *desc1=NULL, *desc2=NULL; 
	CvSURFParams params = cvSURFParams(500, 1);
	cvCvtColor(frame1, grayimage, CV_RGB2GRAY);
	cvExtractSURF( grayimage, NULL, &kp1, &desc1, storage, params );
	cvCvtColor(frame2, grayimage, CV_RGB2GRAY);
	cvExtractSURF( grayimage, NULL, &kp2, &desc2, storage, params );

	std::vector<int> ptpairs;
#ifdef USE_FLANN
	// Using approximate nearest neighbor search
	flannFindPairs( kp1, desc1, kp2, desc2, ptpairs );
#else
	findPairs( kp1, desc1, kp2, desc2, ptpairs );
#endif

	drawSurfResult(frame1, kp1, CV_RGB(255,255,255));
	drawSurfResult(frame2, kp2, CV_RGB(255,255,255));

	int pl = ptpairs.size()/2;
	CvMat *points1 = cvCreateMat(pl,2,CV_32F), *points2 = cvCreateMat(pl,2,CV_32F);
	for(int i=0;i<pl;i++)
	{
		CvSURFPoint* r = (CvSURFPoint*)cvGetSeqElem( kp1, ptpairs[2*i] );
		CV_MAT_ELEM(*points1,float,i,0) = r->pt.x;
		CV_MAT_ELEM(*points1,float,i,1) = r->pt.y;
		r = (CvSURFPoint*)cvGetSeqElem( kp2, ptpairs[2*i+1] );
		CV_MAT_ELEM(*points2,float,i,0) = r->pt.x;
		CV_MAT_ELEM(*points2,float,i,1) = r->pt.y;
	}

	cvFindHomography( points1, points2, homography,CV_FM_RANSAC,1.0);
	cvReleaseMemStorage(&storage);
	cvReleaseImage(&grayimage);
	cvReleaseMat(&points1);
	cvReleaseMat(&points2);
	//cvWarpPerspective(frame1, frame2, hmat);
}
