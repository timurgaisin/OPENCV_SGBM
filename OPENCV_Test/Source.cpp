/*
*  stereo_match.cpp
*  calibration
*
*  Created by Victor  Eruhimov on 1/18/10.
*  Updated by Timur Gaisin :)
*  Copyright 2010 Argus Corp. All rights reserved.
*
*/
#include "opencv2/opencv.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/core/core.hpp"
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <omp.h>

using namespace std;
using namespace cv;


/*h, w = imgL.shape[:2]
f = 0.8*w                          # guess for focal length
Q = np.float32([[1, 0, 0, -0.5*w],
[0, -1, 0, 0.5*h], # turn points 180 deg around x - axis,
[0, 0, 0, -f], # so that y - axis looks up
[0, 0, 1, 0]])*/
Mat makeQMatrix (double w, double h)
{
	double f = 0.8*w;
	Mat Q = Mat::eye(4, 4, CV_64F);
	Q.at<double>(0, 0) = 1.0;
	Q.at<double>(0, 1) = 0.0;
	Q.at<double>(0, 2) = 0.0;
	Q.at<double>(0, 3) = -0.5*w; //cx
	Q.at<double>(1, 0) = 0.0;
	Q.at<double>(1, 1) = -1.0;
	Q.at<double>(1, 2) = 0.0;
	Q.at<double>(1, 3) = 0.5*h;  //cy
	Q.at<double>(2, 0) = 0.0;
	Q.at<double>(2, 1) = 0.0;
	Q.at<double>(2, 2) = 0.0;
	Q.at<double>(2, 3) = -f;	 //Focal
	Q.at<double>(3, 0) = 0.0;
	Q.at<double>(3, 1) = 0.0;
	Q.at<double>(3, 2) = 1.0;    //BaseLine
	Q.at<double>(3, 3) = 0.0;    //cx - cx'*/
	
	return Q;
}

/*Mat makeQMatrix(Point2d image_center, double focal_length, double baseline)
{
	Mat Q = Mat::eye(4, 4, CV_64F);
	Q.at<double>(0, 3) = -image_center.x;
	Q.at<double>(1, 3) = -image_center.y;
	Q.at<double>(2, 3) = focal_length;
	Q.at<double>(3, 3) = 0.0;
	Q.at<double>(2, 2) = 0.0;
	Q.at<double>(3, 2) = 1.0 / baseline;

	return Q;
}*/

void write_ply(std::string fn, Mat verts, Mat colors)
{
	const double max_z = 1.0e4;
	std::string str;
  std::string mesh_filename = "mesh.txt";
	std::ofstream mesh_file(mesh_filename);
	int x2 = 0;
	//verts = np.hstack([verts, colors]);
	int x1 = 0;
	int k_min = 0;
	for (int y = 0; y < verts.rows; y++) {
		for (int x = 0; x < verts.cols; x++) {
			//fout << verts.dims << endl;
			Vec3f point = verts.at<Vec3f>(y, x);
			if (fabs(point[2] - max_z) < FLT_EPSILON || fabs(point[2]) > max_z) {
				for (int c = 0; c < 3; c++) {
					if ((x1 + 1) < 3 * colors.cols) {
						x1++;
					}
					else {
						x1 = 0;
					}
					if (x1 == 0) {
						x2++;
					}
				}
				k_min++;
				continue;
			}
      			mesh_file << point[0] << " " << point[1] << " " << point[2] << " ";
			for (int c = 0; c < 3; c++) {
				if (x1 < 3*colors.cols && x2 < colors.rows) {
					int point_2 = colors.at<uchar>(x2, x1);
					if (c < 2) {
              					mesh_file << point_2 << " ";
					}
					else {
              					mesh_file << point_2 << " " << endl;
					}
				}
				if ((x1 + 1) < 3*colors.cols) {
					x1++;
				}
				else {
					x1 = 0;
				}
				if (x1 == 0) {
					x2++;
				}
			}
			//if (fabs(point[2] - max_z) < FLT_EPSILON || fabs(point[2]) > max_z) continue;
			//fprintf(fp, "%f %f %f\n", point[0], point[1], point[2]);
		}
	}
  	mesh_file.close();

	std::ofstream model_File(fn);
	std::string ply_header = "ply \n";
	ply_header += "format ascii 1.0 \n";
	ply_header += "element vertex " + std::to_string(verts.rows*verts.cols-k_min) + "\n";
	ply_header += "property float x \n";
	ply_header += "property float y \n";
	ply_header += "property float z \n";
	ply_header += "property uchar red \n";
	ply_header += "property uchar green \n";
	ply_header += "property uchar blue \n";
	ply_header += "end_header\n";
  	model_File << ply_header;
	std::ifstream color_file(mesh_filename);
	std::string line;
	while (!color_file.eof()) {
		getline(color_file, line);
    		model_File << line << endl;
	}
  	model_File.close();

  	remove("mesh.txt");
}
	

static void print_help()
{
	printf("\nDemo stereo matching converting L and R images into disparity and point clouds\n");
	printf("\nUsage: stereo_match <left_image> <right_image>  [--blocksize=<block_size>]\n"
		"[--max-disparity=<max_disparity>] [--scale=scale_factor>] \n"
		"[--no-display]\n");
}

static void saveXYZ(const char* filename, const Mat& mat)
{
	const double max_z = 1.0e4;
	FILE* fp = fopen(filename, "wt");
	for (int y = 0; y < mat.rows; y++)
	{
		for (int x = 0; x < mat.cols; x++)
		{
			Vec3f point = mat.at<Vec3f>(y, x);
			if (fabs(point[2] - max_z) < FLT_EPSILON || fabs(point[2]) > max_z) continue;
			fprintf(fp, "%f %f %f\n", point[0], point[1], point[2]);
		}
	}
	fclose(fp);
}

int main(int argc, char** argv)
{
	std::string img1_filename = "";
	std::string img2_filename = "";
	std::string intrinsic_filename = "";
	std::string extrinsic_filename = "";
	std::string disparity_filename = "";
	std::string point_cloud_filename = "point_cloud_2.xyz";

	enum { STEREO_BM = 0, STEREO_SGBM = 1, STEREO_HH = 2, STEREO_VAR = 3, STEREO_3WAY = 4 };
	int alg = STEREO_SGBM;
	int SADWindowSize, numberOfDisparities, minDisparity;
	bool no_display;
	float scale;
	
	Ptr<StereoBM> bm = StereoBM::create(16, 9);
	Ptr<StereoSGBM> sgbm = StereoSGBM::create(0, 16, 3);
	cv::CommandLineParser parser(argc, argv,
		"{help h||}{algorithm||}{max-disparity|0|}{min-disparity|0|}{blocksize|0|}{no-display||}{scale|1|}{i||}{e||}{o||}{p||}");
	
	if (parser.has("help"))
	{
		print_help();
		system("pause");
		return 0;
	}
	cout << "loading..." << endl;
	img1_filename = argv[1];
	img2_filename = argv[2];
	
	if (parser.has("algorithm"))
	{
		std::string _alg = parser.get<std::string>("algorithm");
		alg = _alg == "bm" ? STEREO_BM :
			_alg == "sgbm" ? STEREO_SGBM :
			_alg == "hh" ? STEREO_HH :
			_alg == "var" ? STEREO_VAR :
			_alg == "sgbm3way" ? STEREO_3WAY : -1;
	}
	minDisparity = parser.get<int>("min-disparity"); //16 -64 0
	numberOfDisparities = parser.get<int>("max-disparity"); //192-minDisparity 192 16
	SADWindowSize = parser.get<int>("blocksize"); //5 1
	scale = parser.get<float>("scale"); //0.5
	no_display = parser.has("no-display"); //false
	if (parser.has("i"))
		intrinsic_filename = parser.get<std::string>("i");
	if (parser.has("e"))
		extrinsic_filename = parser.get<std::string>("e");
	if (parser.has("o"))
		disparity_filename = parser.get<std::string>("o");
	if (parser.has("p"))
		point_cloud_filename = parser.get<std::string>("p");

	if (!parser.check())
	{
		parser.printErrors();
		return 1;
	}
	if (alg < 0)
	{
		printf("Command-line parameter error: Unknown stereo algorithm\n\n");
		print_help();
		system("pause");
		return -1;
	}
	if (numberOfDisparities < 1 || numberOfDisparities % 16 != 0)
	{
		printf("Command-line parameter error: The max disparity (--maxdisparity=<...>) must be a positive integer divisible by 16\n");
		print_help();
		system("pause");
		return -1;
	}
	if (scale < 0)
	{
		printf("Command-line parameter error: The scale factor (--scale=<...>) must be a positive floating-point number\n");
		system("pause");
		return -1;
	}
	if (SADWindowSize < 1 || SADWindowSize % 2 != 1)
	{
		printf("Command-line parameter error: The block size (--blocksize=<...>) must be a positive odd number\n");
		system("pause");
		return -1;
	}
	if (img1_filename.empty() || img2_filename.empty())
	{
		printf("Command-line parameter error: both left and right images must be specified\n");
		system("pause");
		return -1;
	}
	if ((!intrinsic_filename.empty()) ^ (!extrinsic_filename.empty()))
	{
		printf("Command-line parameter error: either both intrinsic and extrinsic parameters must be specified, or none of them (when the stereo pair is already rectified)\n");
		system("pause");
		return -1;
	}

	//if (extrinsic_filename.empty() && !point_cloud_filename.empty())
	//{
	//	printf("Command-line parameter error: extrinsic and intrinsic parameters must be specified to compute the point cloud\n");
	//	system("pause");
	//	return -1;
	//}

	int color_mode = alg == STEREO_BM ? 0 : -1;
	Mat img1 = imread(img1_filename, color_mode);
	Mat img2 = imread(img2_filename, color_mode);
	Mat color;
	
	if (img1.empty())
	{
		printf("Command-line parameter error: could not load the first input image file\n");
		system("pause");
		return -1;
	}
	if (img2.empty())
	{
		printf("Command-line parameter error: could not load the second input image file\n");
		system("pause");
		return -1;
	}

	if (scale != 1.f)
	{
		Mat temp1, temp2;
		int method = scale < 1 ? INTER_AREA : INTER_CUBIC;
		resize(img1, temp1, Size(), scale, scale, method);
		img1 = temp1;
		resize(img2, temp2, Size(), scale, scale, method);
		img2 = temp2;
	}

	Size img_size = img1.size();

	Rect roi1, roi2;
	Mat Q;

	

	if (!intrinsic_filename.empty())
	{
		// reading intrinsic parameters
		FileStorage fs(intrinsic_filename, FileStorage::READ);
		if (!fs.isOpened())
		{
			printf("Failed to open file %s\n", intrinsic_filename.c_str());
			system("pause");
			return -1;
		}

		Mat M1, D1, M2, D2;
		fs["M1"] >> M1;
		fs["D1"] >> D1;
		fs["M2"] >> M2;
		fs["D2"] >> D2;

		M1 *= scale;
		M2 *= scale;

		fs.open(extrinsic_filename, FileStorage::READ);
		if (!fs.isOpened())
		{
			printf("Failed to open file %s\n", extrinsic_filename.c_str());
			system("pause");
			return -1;
		}

		Mat R, T, R1, P1, R2, P2;
		fs["R"] >> R;
		fs["T"] >> T;

		stereoRectify(M1, D1, M2, D2, img_size, R, T, R1, R2, P1, P2, Q, CALIB_ZERO_DISPARITY, -1, img_size, &roi1, &roi2);

		Mat map11, map12, map21, map22;
		initUndistortRectifyMap(M1, D1, R1, P1, img_size, CV_16SC2, map11, map12);
		initUndistortRectifyMap(M2, D2, R2, P2, img_size, CV_16SC2, map21, map22);

		Mat img1r, img2r;
		remap(img1, img1r, map11, map12, INTER_LINEAR);
		remap(img2, img2r, map21, map22, INTER_LINEAR);

		img1 = img1r;
		img2 = img2r;
	}

	numberOfDisparities = numberOfDisparities > 0 ? numberOfDisparities : ((img_size.width / 8) + 15) & -16;

	bm->setROI1(roi1);
	bm->setROI2(roi2);
	bm->setPreFilterCap(31);
	bm->setBlockSize(SADWindowSize > 0 ? SADWindowSize : 9);
	bm->setMinDisparity(0);
	bm->setNumDisparities(numberOfDisparities);
	bm->setTextureThreshold(10);
	bm->setUniquenessRatio(15);
	bm->setSpeckleWindowSize(100);
	bm->setSpeckleRange(32);
	bm->setDisp12MaxDiff(1);

	sgbm->setPreFilterCap(63);
	int sgbmWinSize = SADWindowSize > 0 ? SADWindowSize : 3;
	sgbm->setBlockSize(sgbmWinSize);

	int cn = img1.channels();

	sgbm->setP1(8 * cn*sgbmWinSize*sgbmWinSize);
	sgbm->setP2(32 * cn*sgbmWinSize*sgbmWinSize);
	sgbm->setMinDisparity(minDisparity);//-64//0
	sgbm->setNumDisparities(numberOfDisparities);
	sgbm->setUniquenessRatio(10);//1//10
	sgbm->setSpeckleWindowSize(100);//150//100
	sgbm->setSpeckleRange(32);//2//32
	sgbm->setDisp12MaxDiff(1);//10//1
	if (alg == STEREO_HH)
		sgbm->setMode(StereoSGBM::MODE_HH);
	else if (alg == STEREO_SGBM)
		sgbm->setMode(StereoSGBM::MODE_HH);
	else if (alg == STEREO_3WAY)
		sgbm->setMode(StereoSGBM::MODE_SGBM_3WAY);

	Mat disp, disp8;
	//Mat img1p, img2p, dispp;
	//copyMakeBorder(img1, img1p, 0, 0, numberOfDisparities, 0, IPL_BORDER_REPLICATE);
	//copyMakeBorder(img2, img2p, 0, 0, numberOfDisparities, 0, IPL_BORDER_REPLICATE);

	int64 t = getTickCount();
	if (alg == STEREO_BM)
		bm->compute(img1, img2, disp);
	else if (alg == STEREO_SGBM || alg == STEREO_HH || alg == STEREO_3WAY)
		sgbm->compute(img1, img2, disp);
	t = getTickCount() - t;
	printf("Time elapsed: %fms\n", t * 1000 / getTickFrequency());

	//disp = dispp.colRange(numberOfDisparities, img1p.cols);
	if (alg != STEREO_VAR)
		disp.convertTo(disp8, CV_8U, 255 / (numberOfDisparities*16.));
	else
		disp.convertTo(disp8, CV_8U);
	if (!no_display)
	{
		namedWindow("left", 1);
		imshow("left", img1);
		namedWindow("right", 1);
		imshow("right", img2);
		namedWindow("disparity", 0);
		imshow("disparity", disp8);
		printf("press any key to continue...");
		fflush(stdout);
		waitKey();
		printf("\n");
	}

	if (!disparity_filename.empty())
		imwrite(disparity_filename, disp8);

	//if (!point_cloud_filename.empty())
	//{
		printf("storing the point cloud...");
		fflush(stdout);

		//(2)make Q matrix and reproject pixels into 3D space
		const double focal_length = 598.57;
		const double baseline = 14.0;
		//Q = makeQMatrix(Point2d((img1.cols - 1.0) / 2.0, (img1.rows - 1.0) / 2.0), focal_length, baseline * 16);
		Q = makeQMatrix(img1.cols, img1.rows);

		Mat xyz;
		reprojectImageTo3D(disp, xyz, Q, true);
		cvtColor(img1, color, COLOR_BGR2RGB);

		Mat out_points = xyz;//[mask];
		Mat out_colors = color;//[mask];
		write_ply("model.ply", out_points, out_colors);
		saveXYZ(point_cloud_filename.c_str(), xyz);
		printf("\n");
		
	//}
	system("pause");
	return 0;
}
