#ifndef LFREFOCUSER_H
#define LFREFOCUSER_H
#include <QtCore/qcontainerfwd.h>
#include <QtCore/qobjectdefs.h>
#include <QtCore/qtmetamacros.h>
#include <opencv2/core/hal/interface.h>

#include <QObject>
#include <QThread>
#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>

#include "lfdata.h"

namespace LFRefocus {
class Core {
   public:
	explicit Core() : lf() {}
	void init(const LightFieldPtr& ptr);
	void refocus(float alpha, int crop);
	void updateLF(const LightFieldPtr& ptr);
	void setGPU(bool isGPU);

	bool	getGpu() const { return _isGPU; }
	cv::Mat getRefocusedImage() const { return _refocusedImage; }

	LightFieldPtr lf;

   private:
	bool	 _isGPU = false;
	int		 _views, _len, _center, _type;
	cv::Mat	 _xmap, _ymap, _refocusedImage;
	cv::UMat _xmap_gpu, _ymap_gpu;
	cv::Size _size;
};
class Worker : public QObject {
	Q_OBJECT
   public:
	explicit Worker(QObject* parent = nullptr);
   public slots:
	void printThreadId();
	void setGpu(bool isGPU);
	void refocus(float alpha, int crop);
	void lfUpdated(const LightFieldPtr& ptr);
   signals:
	void requestUpdateSAI(const cv::Mat& image);

   private:
	std::unique_ptr<Core> _core;
};
} // namespace LFRefocus
#endif