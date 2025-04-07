#ifndef LFREFOCUSER_H
#define LFREFOCUSER_H
#include <QtCore/qcontainerfwd.h>
#include <QtCore/qobject.h>
#include <QtCore/qobjectdefs.h>
#include <QtCore/qstring.h>
#include <QtCore/qthread.h>
#include <QtCore/qtmetamacros.h>
#include <QtCore/qvariant.h>

#include <QObject>
#include <QThread>
#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <utility>
#include <vector>

#include "worker_interface.h"

namespace LFRefocus {
class Core {
   public:
	Core(const std::vector<cv::Mat>& src);
	~Core();

	void init(const std::vector<cv::Mat>& src);
	void refocus(float alpha, int offset);
	void setLF(const std::vector<cv::Mat>& src);
	void setGPU(bool isGPU);

	bool	getGPU();
	cv::Mat getRefocusedImage() const { return _refocusedImage; }

   private:
	std::vector<cv::Mat>				   _lf;
	std::unique_ptr<std::vector<cv::UMat>> _lf_gpu;
	bool								   _isGPU = false, isStop = false;
	int									   _views, _len, _center, _type;
	cv::Mat								   _xmap, _ymap, _refocusedImage;
	cv::UMat							   _xmap_gpu, _ymap_gpu;
	cv::Size							   _size;
};
class Worker : public WorkerInterface {
	Q_OBJECT
   public:
	Worker(const std::vector<cv::Mat>& src);
	Q_INVOKABLE void refocus(float alpha, int offset);
	Q_INVOKABLE void setGpu(bool enable);
	Q_INVOKABLE bool getGpu();

	template <typename... Args>
	void executeAlgorithm(void (Core::*method)(Args...), Args&&... args) {
		invokeAsync([this, method,
					 argsTuple = std::make_tuple(
						 std::forward<Args>(args)...)]() mutable {
			std::apply(
				[this, method](auto&&... unpackedArgs) {
					(_core.get()->*method)(
						std::forward<decltype(unpackedArgs)>(unpackedArgs)...);
				},
				std::move(argsTuple));
			emit operationCompleted("algorithmExecuted", QVariant(true));
		});
	}

   signals:
	void operationCompleted(QString type, QVariant result);
	// void refocusCompleted(std::chrono::duration<double> elapsed);

	//    private:
   public:
	std::unique_ptr<Core> _core;
};
} // namespace LFRefocus

class QLFRefocuser : public QObject {
	Q_OBJECT
   public:
	QLFRefocuser(const std::vector<cv::Mat>& src, QObject* parent = nullptr);
	~QLFRefocuser();
	// template <typename... Args>
	// void execute(const QString& operation, Args... args) {
	// 	_worker->invoke(operation.toStdString().c_str(),
	// 					std::forward<Args>(args)...);
	// }
	template <typename... Args>
	void execute(const QString& opName,
				 void (LFRefocus::Core::*method)(Args...), Args&&... args) {
		_worker->invokeAsync([this, opName, method,
							  argsTuple = std::make_tuple(
								  std::forward<Args>(args)...)]() mutable {
			std::apply(
				[this, method](auto&&... unpackedArgs) {
					(_worker->_core.get()->*method)(
						std::forward<decltype(unpackedArgs)>(unpackedArgs)...);
				},
				std::move(argsTuple));
			emit _worker->operationCompleted(opName, QVariant(true));
		});
	}
	// void execute(void (LFRefocus::Core::*method)(Args...), Args&&... args) {
	// 	_worker->executeAlgorithm(method, std::forward<Args>(args)...);
	// }
   signals:
	void resultReady(QString operation, QVariant result);

   private:
	QThread*		   _thread;
	LFRefocus::Worker* _worker;
};

#endif