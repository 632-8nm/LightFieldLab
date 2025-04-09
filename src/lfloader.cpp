#include "lfloader.h"

#include <QtWidgets/qwidget.h>

#include <filesystem>
#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <vector>

// LFLoader::LFLoader(QWidget* parent) : QWidget(parent) {}
// LFLoader::~LFLoader() {}

// void LFLoader::loadSAI(std::string path, bool isRGB) {
// 	LF.clear();
// 	LF_float32.clear();
// 	std::vector<std::string> filenames;

// 	for (const auto& entry : std::filesystem::directory_iterator(path)) {
// 		if (entry.is_regular_file()) {
// 			filenames.push_back(
// 				entry.path().filename().string()); // 只保存文件名
// 		}
// 	}

// 	// 排序
// 	std::sort(filenames.begin(), filenames.end());

// 	// 输出
// 	for (const auto& name : filenames) {
// 		// std::cout << name << std::endl;
// 		cv::Mat img, img_float32;
// 		if (isRGB) {
// 			img = cv::imread(path + name, cv::IMREAD_COLOR);
// 		} else {
// 			img = cv::imread(path + name, cv::IMREAD_GRAYSCALE);
// 		}

// 		img.convertTo(img_float32, CV_32FC(img.channels()));
// 		LF.push_back(img);
// 		LF_float32.push_back(img_float32);
// 	}
// }
namespace LFLoader {
Core::Core() {}
Core::~Core() {}

void Core::loadSAI(const std::string& path, const bool& isRGB) {
	_lf.clear();
	_lf_float32.clear();
	std::vector<std::string> filenames;

	for (const auto& entry : std::filesystem::directory_iterator(path)) {
		if (entry.is_regular_file()) {
			filenames.push_back(
				entry.path().filename().string()); // 只保存文件名
		}
	}

	// 排序
	std::sort(filenames.begin(), filenames.end());

	// 输出
	for (const auto& name : filenames) {
		// std::cout << name << std::endl;
		cv::Mat		img, img_float32;
		std::string filename = path + "/" + name;
		std::cout << filename << std::endl;
		if (isRGB) {
			img = cv::imread(filename, cv::IMREAD_COLOR);
		} else {
			img = cv::imread(filename, cv::IMREAD_GRAYSCALE);
		}

		img.convertTo(img_float32, CV_32FC(img.channels()));
		_lf.push_back(img);
		_lf_float32.push_back(img_float32);
	}
}
} // namespace LFLoader