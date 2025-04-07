#include "lfrefocuser.h"

#include <QtCore/qmetatype.h>
#include <QtCore/qobject.h>
#include <QtCore/qobjectdefs.h>
#include <QtCore/qthread.h>
#include <QtCore/qtmetamacros.h>
#include <QtCore/qvariant.h>
#include <QtWidgets/qwidget.h>
#include <opencv2/core/hal/interface.h>

#include <cmath>
#include <cstddef>
#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/core/base.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>
namespace LFRefocus {
Core::Core(const std::vector<cv::Mat>& src) : _isGPU(false) { setLF(src); }
Core::~Core() {}
void Core::init(const std::vector<cv::Mat>& src) {
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
void Core::setLF(const std::vector<cv::Mat>& src) {
	_lf = src;
	if (src.size() == _views && src[0].type() == _type
		&& src[0].size() == _size) {
		return;
	}
	init(src);
}
void Core::setGPU(bool isGPU) {
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
bool Core::getGPU() { return _isGPU; }
void Core::refocus(float alpha, int offset) {
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
Worker::Worker(const std::vector<cv::Mat>& src) : _core(nullptr) {
	_core = std::make_unique<Core>(src);
}
void Worker::refocus(float alpha, int offset) {
	auto start = std::chrono::high_resolution_clock::now();
	_core->refocus(alpha, offset);
	auto end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsed = end - start;
	emit operationCompleted("refocus", QVariant::fromValue(elapsed));
	// emit refocusCompleted(elapsed);
}
void Worker::setGpu(bool enable) {
	_core->setGPU(enable);
	emit operationCompleted("setGpu", enable);
}

// getGpu 实现
bool Worker::getGpu() {
	bool status = _core->getGPU();
	emit operationCompleted("getGpu", status);
	return status;
}

} // namespace LFRefocus

QLFRefocuser::QLFRefocuser(const std::vector<cv::Mat>& src, QObject* parent)
	: QObject(parent), _thread(nullptr), _worker(nullptr) {
	_worker = new LFRefocus::Worker(src);
	_thread = new QThread(this);
	_worker->moveToThread(_thread);
	_thread->start();

	connect(_worker, &LFRefocus::Worker::operationCompleted, this,
			&QLFRefocuser::resultReady);
	qRegisterMetaType<QVariant>("QVariant");
}
QLFRefocuser::~QLFRefocuser() {
	if (_thread) {
		_thread->quit();
		_thread->wait();
		delete _thread;
		_thread = nullptr;
	}
	if (_worker) {
		delete _worker;
		_worker = nullptr;
	}
}
