#include <stdlib.h>
#include <stdio.h>
#include <iostream>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/gpu/gpu.hpp>
#include <opencv2/contrib/contrib.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;
using namespace cv::gpu;

//Determination distance to face
//res - gorizontal resolution, size - width of rectangle with face
int Distance(int res, int size)
{
	double x = 0.0;
	double dist = 0.0;

	x = (double)size/(double)res;
	dist = 6889.25*x*x - 2962.42*x + 366.82;  //interpolation polynom

	return (int)dist;
}

int main(int argc, char* argv[])
{
	bool useGPU = true; //flag use/not use GPU
	bool multi = true; //detect multi/one face (the largest)
	string cascadeName = "haarcascade_frontalface_alt.xml"; //Haar cascade for detector

	//Variables for frames in memory
	Mat frame_cpu, frameGray_cpu, facesBuf_cpu;
	GpuMat frame_gpu, frameGray_gpu, facesBuf_gpu;
	Rect* faces_gpu;
	std::vector<Rect> faces_cpu;

	//Detecting cascades
	CascadeClassifier_GPU cascade_gpu;
	CascadeClassifier cascade_cpu;

	int numDetect = 0; //detections counter
	double time; //var for calculating FPS
	char fps[4];
	char out[4]; //vars for text output to window
	int resX, resY; //screen resolution
	int dist; //distance to object

	//when use GPU it is possibility to draw rectangles inplace
	cascade_gpu.visualizeInPlace = true;

	//check for CUDA-capable devices
	if (getCudaEnabledDeviceCount() == 0)
	{
		printf("[e] no GPU found or the library is compiled without GPU support\n\n");
	}
	printShortCudaDeviceInfo(getDevice());

	//loading Haar cascades for CPU and GPU
	if ((!cascade_gpu.load(cascadeName)) || (!cascade_cpu.load(cascadeName)))
	{
		printf("[e] loading cascade failed\n\n");
	}

	//check for command-line arguments with necessary resolution
	if (argc == 3)
	{
		//loading new value
		resX = atoi(argv[1]);
		resY = atoi(argv[2]);
	}
	else
	{
		//setting default value if there are no arguments
		resX = 800;
		resY = 450;
	}

	//connecting to camera (any device)
	CvCapture* capture = cvCreateCameraCapture(CV_CAP_ANY);
	assert( capture );

	//set resolution
	cvSetCaptureProperty(capture, CV_CAP_PROP_FRAME_WIDTH, resX); 
	cvSetCaptureProperty(capture, CV_CAP_PROP_FRAME_HEIGHT, resY); 

	//creating window for images
	cvNamedWindow("capture", CV_WINDOW_AUTOSIZE);

	printf("[i] press ESC for quit!\n");
	printf("[i] press 'G' or 'g' for changing CPU/GPU mode\n");
	printf("[i] press 'M' or 'm' for changing one/multi mode\n");

	while(true)
	{
		//receiveng next frame from camera
		frame_cpu = cvQueryFrame( capture );

		//starting "fps" measuring
		TickMeter tm;
		tm.start();

		if (useGPU)
		{
			//loading frame to GPU memory
			frame_gpu.upload(frame_cpu);

			//changing color state to grayscale
			cvtColor(frame_gpu, frameGray_gpu, CV_BGR2GRAY);

			//equalizing the histogram (not necessary, so commented)
			//equalizeHist(frameGray_gpu, frameGray_gpu);

			//setting detect mode (multi/one)
			cascade_gpu.findLargestObject = !multi;

			//launching detection on GPU
			numDetect = cascade_gpu.detectMultiScale( frameGray_gpu, facesBuf_gpu, 1.2, 4);

			//another way to download frame from GPU with detected rectangles (alternative, so commented)
			//cvtColor(frameGray_gpu, frame_gpu, CV_GRAY2BGR);
			//frame_gpu.download(frame_cpu);

			//downloading only rectangles from GPU
			facesBuf_gpu.colRange(0, numDetect).download(facesBuf_cpu);
			faces_gpu = facesBuf_cpu.ptr<Rect>();

			//visualizing received rectangles
			for(int i = 0; i < numDetect; ++i)
			{
				//drawing rectangle around detected face
				rectangle(frame_cpu, faces_gpu[i], CV_RGB(255, 0, 0), 2); 

				//calculating distance to object and printing it into the rectangle
				dist = Distance(resX, faces_gpu[i].width);
				itoa(dist, out, 10);
				putText(frame_cpu, out, Point(faces_gpu[i].x + 2, faces_gpu[i].y + 17), FONT_HERSHEY_PLAIN, 1.2, CV_RGB(255, 0, 0), 2);
			}
		} 
		else
		{
			//changing color state to grayscale
			cvtColor(frame_cpu, frameGray_cpu, CV_BGR2GRAY);

			//equalizing the histogram (not necessary, so commented)
			//equalizeHist(frameGray_cpu, frameGray_cpu);

			//calculating the size of cascade classifier (now using hardcode parameter for size (50,50) for better perfomance)
			//Size minSize = cascade_gpu.getClassifierSize();

			//launching detection on CPU
			cascade_cpu.detectMultiScale(frameGray_cpu, faces_cpu, 1.2, 4, (!multi ? CV_HAAR_FIND_BIGGEST_OBJECT : 0), Size(50, 50));
			numDetect = (int)faces_cpu.size();

			//visualizing calculated rectangles
			for(int i = 0; i < numDetect; ++i)
			{
				//drawing rectangle around detected face
				rectangle(frame_cpu, faces_cpu[i], CV_RGB(0, 0, 255), 2);

				//calculating distance to object and printing it into the rectangle
				dist = Distance(resX, faces_cpu[i].width);
				itoa(dist, out, 10);
				putText(frame_cpu, out, Point(faces_cpu[i].x + 2, faces_cpu[i].y + 17), FONT_HERSHEY_PLAIN, 1.2, CV_RGB(0, 0, 255), 2);
			}
		}
		
		//finish for fps counter, printing to window
		tm.stop();
		time = 1000.0 / tm.getTimeMilli();
		itoa(time, fps, 10);
		putText(frame_cpu, fps, Point(10, 30), FONT_HERSHEY_DUPLEX, 1, CV_RGB(0, 255, 0), 1);

		//displaying result
		imshow("capture", frame_cpu);

		//handling pressing keys
		char c = cvWaitKey(1);
		if (c == 27) 
		{ //ESC
			break;
		}
		else if (c == 71 || c == 103)
		{
			//Press "g" for changing CPU/GPU mode
			useGPU = !useGPU;
		}
		else if (c == 77 || c == 109)
		{
			//Press "m" for changing one/multi mode
			multi = !multi;
		}
	}

	//cleaning
	cvReleaseCapture( &capture );
	cvDestroyWindow("capture");
	return 0;
}