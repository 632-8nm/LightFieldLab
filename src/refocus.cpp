#include "refocus.h"

#include <QtCore/qthread.h>
#include <opencv2/core/hal/interface.h>

#include <cmath>
#include <opencv2/core.hpp>
#include <opencv2/core/base.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>

Refocus::Refocus(QObject* parent) : QObject(parent), _isGPU(false) {
	// Constructor implementation
}
Refocus::Refocus(const std::vector<cv::Mat>& src, QObject* parent)
	: QObject(parent), _isGPU(false) {
	// Constructor implementation
	setLF(src);
}
Refocus::~Refocus() {
	// Destructor implementation
}
void Refocus::para_init(const std::vector<cv::Mat>& src) {
	_views = _lf.size();
	_len   = static_cast<int>(std::sqrt(_views));
	_size  = _lf[0].size();
	_type  = _lf[0].type();

	_xmap = cv::Mat(_size.height, 1, CV_32FC1);
	_ymap = cv::Mat(1, _size.width, CV_32FC1);
	for (int x = 0; x < _size.height; x++) {
		_xmap.at<float>(x, 0) = static_cast<float>(x);
	}
	for (int y = 0; y < _size.width; y++) {
		_ymap.at<float>(0, y) = static_cast<float>(y);
	}
	_xmap = cv::repeat(_xmap, 1, _size.width);
	_ymap = cv::repeat(_ymap, _size.height, 1);
}
void Refocus::setLF(const std::vector<cv::Mat>& src) {
	_lf = src; // TODO 深拷贝
	if (src.size() == _views && src[0].type() == _type
		&& src[0].size() == _size) {
		return;
	}
	para_init(src);
}
void Refocus::setGPU(bool isGPU) {
	if (_isGPU == isGPU) {
		return; // 状态未变化，无需操作
	}

	_isGPU = isGPU;

	if (_isGPU) {
		// GPU 模式：分配并复制数据
		_lf_gpu = std::make_unique<std::vector<cv::UMat>>(_lf.size());
		for (size_t i = 0; i < _lf.size(); ++i) {
			(*_lf_gpu)[i] = _lf[i].getUMat(cv::ACCESS_READ);
		}
		_xmap_gpu = _xmap.getUMat(cv::ACCESS_READ);
		_ymap_gpu = _ymap.getUMat(cv::ACCESS_READ);
	} else {
		// CPU 模式：释放 GPU 资源
		_lf_gpu.reset(); // 自动释放 vector
		_xmap_gpu.release();
		_ymap_gpu.release();
	}
}
void Refocus::refocus(float alpha, int offset) {
	float	 factor	 = 1.0f - 1.0f / alpha;
	int		 divisor = (_len - 2 * offset) * (_len - 2 * offset);
	cv::Mat	 temp, sum, xq, yq;
	cv::UMat temp_gpu, sum_gpu, xq_gpu, yq_gpu;
	sum				= cv::Mat(_size, _type, cv::Scalar(0));
	_refocusedImage = cv::Mat(_size, _type, cv::Scalar(0)); // 清除结果

	if (_isGPU) {
		sum_gpu = cv::UMat(_size, _type, cv::Scalar(0));
	}

	for (int i = 0; i < _lf.size(); i++) {
		int row = i / _len;
		int col = i % _len;
		if (row < offset || col < offset || row >= _len - offset
			|| col >= _len - offset) {
			continue;
		}
		if (_isGPU) {
			cv::add(_ymap_gpu, factor * (col - _center), yq_gpu);
			cv::add(_xmap_gpu, factor * (row - _center), xq_gpu);
			cv::remap((*_lf_gpu)[i], temp_gpu, yq_gpu, xq_gpu, cv::INTER_LINEAR,
					  cv::BORDER_REPLICATE);
			cv::add(temp_gpu, sum_gpu, sum_gpu);
		} else {
			cv::add(_ymap, factor * (col - _center), yq);
			cv::add(_xmap, factor * (row - _center), xq);
			cv::remap(_lf[i], temp, yq, xq, cv::INTER_LINEAR,
					  cv::BORDER_REPLICATE);
			cv::add(temp, sum, sum);
		}
	}
	if (_isGPU) {
		cv::divide(sum_gpu, divisor, sum_gpu);
		_refocusedImage = sum_gpu.getMat(cv::ACCESS_READ);
	} else {
		cv::divide(sum, divisor, _refocusedImage);
	}
}
void Refocus::work_test() {
	while (true) {
		auto start = std::chrono::high_resolution_clock::now();
		refocus(1.5, 2);
		auto end = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> elapsed = end - start;
		emit						  refocusFinished(elapsed);
	}
}