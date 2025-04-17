#include "lfrefocuser.h"

#include <QtCore/qlogging.h>
#include <QtCore/qmetatype.h>
#include <QtCore/qobject.h>
#include <QtCore/qobjectdefs.h>
#include <QtCore/qthread.h>
#include <QtCore/qtmetamacros.h>
#include <QtCore/qvariant.h>
#include <QtWidgets/qwidget.h>
#include <opencv2/core/hal/interface.h>

#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/core/base.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/ocl.hpp> // 确保包含头文件
#include <opencv2/core/types.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>

namespace LFRefocus {
void Core::init(const LightFieldPtr& ptr) {
	_size	= ptr->data[0].size();
	_type	= ptr->data[0].type();
	_center = (1 + ptr->rows) / 2 - 1;

	_xmap = cv::Mat(ptr->height, 1, CV_32FC1);
	_ymap = cv::Mat(1, ptr->width, CV_32FC1);
	for (int x = 0; x < ptr->height; x++) {
		_xmap.at<float>(x, 0) = static_cast<float>(x);
	}
	for (int y = 0; y < ptr->width; y++) {
		_ymap.at<float>(0, y) = static_cast<float>(y);
	}
	_xmap = cv::repeat(_xmap, 1, ptr->width);
	_ymap = cv::repeat(_ymap, ptr->height, 1);
}

void Core::updateLF(const LightFieldPtr& ptr) {
	lf = ptr;
	init(ptr);
}
void Core::setGPU(bool isGPU) {
	_isGPU = isGPU;

	if (_isGPU) {
		_xmap_gpu = _xmap.getUMat(cv::ACCESS_READ);
		_ymap_gpu = _ymap.getUMat(cv::ACCESS_READ);
	} else {
		_xmap_gpu.release();
		_ymap_gpu.release();
	}
}
void Core::refocus(float alpha, int crop) {
	if (lf == nullptr) {
		qDebug() << "lf is nullptr!\n";
		return;
	}

	float	 factor	 = 1.0f - 1.0f / alpha;
	int		 divisor = (lf->rows - 2 * crop) * (lf->cols - 2 * crop);
	cv::Mat	 temp, sum, xq, yq;
	cv::UMat temp_gpu, sum_gpu, xq_gpu, yq_gpu;
	sum				= cv::Mat(_size, _type, cv::Scalar(0));
	_refocusedImage = cv::Mat(_size, _type, cv::Scalar(0)); // 清除结果

	if (_isGPU) {
		sum_gpu = cv::UMat(_size, _type, cv::Scalar(0));
	}

	for (int i = 0; i < lf->size; i++) {
		int row = i / lf->rows;
		int col = i % lf->cols;
		if (row < crop || col < crop || row >= lf->rows - crop
			|| col >= lf->cols - crop) {
			continue;
		}
		if (_isGPU) {
			cv::add(_ymap_gpu, factor * (col - _center), yq_gpu);
			cv::add(_xmap_gpu, factor * (row - _center), xq_gpu);
			cv::remap(lf->data_gpu[i], temp_gpu, yq_gpu, xq_gpu,
					  cv::INTER_LINEAR, cv::BORDER_REPLICATE);
			cv::add(temp_gpu, sum_gpu, sum_gpu);
		} else {
			cv::add(_ymap, factor * (col - _center), yq);
			cv::add(_xmap, factor * (row - _center), xq);
			cv::remap(lf->data[i], temp, yq, xq, cv::INTER_LINEAR,
					  cv::BORDER_REPLICATE);
			cv::add(temp, sum, sum);
		}
	}
	if (_isGPU) {
		cv::divide(sum_gpu, divisor, sum_gpu);
		_refocusedImage = sum_gpu.getMat(cv::ACCESS_READ).clone(); // clone
	} else {
		cv::divide(sum, divisor, _refocusedImage);
	}
}
Worker::Worker(QObject* parent) : QObject(parent) {
	_core = std::make_unique<Core>();
}
void Worker::printThreadId() {
	std::cout << "LFRefocus threadId: " << QThread::currentThreadId()
			  << " = printThreadId called!" << std::endl;
}
void Worker::setGpu(bool isGPU) { _core->setGPU(isGPU); }
void Worker::refocus(float alpha, int crop) {
	_core->refocus(alpha, crop);
	cv::Mat cvImg, cvImg_float;
	cvImg_float = _core->getRefocusedImage();
	cvImg_float.convertTo(cvImg, CV_8UC(cvImg_float.channels()));
	emit requestUpdateSAI(cvImg);
}
void Worker::lfUpdated(const LightFieldPtr& ptr) { _core->updateLF(ptr); }

} // namespace LFRefocus
