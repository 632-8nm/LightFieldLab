#include <opencv2/core/hal/interface.h>

#include <cstring>
#include <filesystem>
#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <vector>

#include "refocus.h"

int main(int argc, char* argv[]) {
	std::cout << "Testing Refocus class" << std::endl;
	std::string path = std::string(argv[1]);

	std::vector<std::string> filenames;

	for (const auto& entry : std::filesystem::directory_iterator(path)) {
		if (entry.is_regular_file()) {
			filenames.push_back(
				entry.path().filename().string()); // 只保存文件名
		}
	}

	// 排序
	std::sort(filenames.begin(), filenames.end());

	std::vector<cv::Mat> LF, LF_float32;
	// 输出
	for (const auto& name : filenames) {
		// std::cout << name << std::endl;
		cv::Mat img, img_float32;
		if (strcmp(argv[2], "gray") == 0) {
			img = cv::imread(path + name, cv::IMREAD_GRAYSCALE);
		} else {
			img = cv::imread(path + name, cv::IMREAD_COLOR);
		}

		img.convertTo(img_float32, CV_32FC(img.channels()));
		LF.push_back(img);
		LF_float32.push_back(img_float32);
	}

	// cv::imshow("center", LF[112]);
	// cv::waitKey(0);

	Refocus* rfc	= new Refocus(nullptr, LF_float32);
	float	 alpha	= 1.5;
	int		 offset = 2;
	// cpu计算
	rfc->setGPU(false);
	auto start = std::chrono::high_resolution_clock::now();
	rfc->refocus(alpha, offset);
	auto end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsed = end - start;
	std::cout << "====== Refocus time: " << elapsed.count()
			  << " seconds ======" << std::endl;
	// cv::Mat refocusedImg = rfc->getRefocusedImage();
	// refocusedImg.convertTo(refocusedImg, CV_8UC(refocusedImg.channels()));
	// cv::imshow("1 cpu", refocusedImg);
	// cv::waitKey(0);

	// gpu计算
	rfc->setGPU(true);
	start = std::chrono::high_resolution_clock::now();
	rfc->refocus(alpha, offset);
	end		= std::chrono::high_resolution_clock::now();
	elapsed = end - start;
	std::cout << "====== Refocus time: " << elapsed.count()
			  << " seconds ======" << std::endl;
	// refocusedImg = rfc->getRefocusedImage();
	// refocusedImg.convertTo(refocusedImg, CV_8UC(refocusedImg.channels()));
	// cv::imshow("2 gpu", refocusedImg);
	//   cv::waitKey(0);

	// cpu计算
	rfc->setGPU(false);
	start = std::chrono::high_resolution_clock::now();
	rfc->refocus(alpha, offset);
	end		= std::chrono::high_resolution_clock::now();
	elapsed = end - start;
	std::cout << "====== Refocus time: " << elapsed.count()
			  << " seconds ======" << std::endl;

	cv::Mat refocusedImg = rfc->getRefocusedImage();
	refocusedImg.convertTo(refocusedImg, CV_8UC(refocusedImg.channels()));
	cv::imshow("3 cpu", refocusedImg);
	cv::waitKey(0);

	delete rfc;
	return 0;
}