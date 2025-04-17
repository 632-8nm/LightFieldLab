#include "lfloader.h"

#include <QtCore/qlogging.h>
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
	if (!lf.empty()) {
		lf.clear();
	}

	std::vector<std::string> filenames;
	for (const auto& entry : std::filesystem::directory_iterator(path)) {
		if (entry.is_regular_file()) {
			filenames.push_back(
				entry.path().filename().string()); // 只保存文件名
		}
	}
	std::sort(filenames.begin(), filenames.end());

	std::vector<cv::Mat> temp;
	for (const auto& name : filenames) {
		cv::Mat		img;
		std::string filename = path + "/" + name;
		// std::cout << filename << std::endl;
		if (isRGB) {
			img = cv::imread(filename, cv::IMREAD_COLOR);
		} else {
			img = cv::imread(filename, cv::IMREAD_GRAYSCALE);
		}
		temp.emplace_back(img);
	}
	lf = LightField(temp);
	std::cout << "Loading finished!" << std::endl;
}
Worker::Worker(QObject* parent) : QObject(parent) {
	_core = std::make_unique<Core>();
}
void Worker::printThreadId() {
	std::cout << "LFLoader threadId: " << QThread::currentThreadId()
			  << " == printThreadId called!" << std::endl;
}
void Worker::load(const QString& path, const bool& isRGB) {
	std::cout << "load called! == LFLoader threadId: "
			  << QThread::currentThreadId() << std::endl;
	_core->load(path.toStdString(), isRGB);
	emit lfUpdated(std::make_shared<LightField>(_core->lf));
};
}; // namespace LFLoader