#include "lfloader.h"

#include <QtWidgets/qwidget.h>

#include <algorithm>
#include <filesystem>
#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <vector>

#include "lfdata.h"

namespace LFLoader {
void Core::load(const std::string& path, const bool& isRGB) {
	if (!_lf.empty() && !_lf_float32.empty()) {
		_lf.clear();
		_lf_float32.clear();
	}

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
	std::vector<cv::Mat> lf, lf_float32;
	for (const auto& name : filenames) {
		cv::Mat		img, img_float32;
		std::string filename = path + "/" + name;
		// std::cout << filename << std::endl;
		if (isRGB) {
			img = cv::imread(filename, cv::IMREAD_COLOR);
		} else {
			img = cv::imread(filename, cv::IMREAD_GRAYSCALE);
		}

		img.convertTo(img_float32, CV_32FC(img.channels()));
		lf.push_back(img);
		lf_float32.push_back(img_float32);
	}
	_lf			= std::move(LightField(lf));
	_lf_float32 = std::move(LightField(lf_float32));
	std::cout << "Loading finished!" << std::endl;
}
} // namespace LFLoader