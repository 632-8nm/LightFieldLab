#ifndef LFLOADER_H
#define LFLOADER_H

#include <QtCore/qobject.h>
#include <QtCore/qtmetamacros.h>

#include <QObject>
#include <QThread>
#include <QWidget>
#include <opencv2/core/mat.hpp>
#include <opencv2/highgui.hpp>
#include <string>
#include <vector>

#include "interface.h"

namespace LFLoader {
class Core {
   public:
	Core();
	~Core();

	void loadSAI(const std::string& path, const bool& isRGB);
	void showCenter() const {
		cv::imshow("center", _lf[112]);
		cv::waitKey();
	}
	std::vector<cv::Mat> getLF() const { return _lf; }
	std::vector<cv::Mat> getLF_float32() const { return _lf_float32; }

   private:
	std::vector<cv::Mat> _lf;
	std::vector<cv::Mat> _lf_float32;
};
class Worker : public WorkerBase<Core> {
	Q_OBJECT
   signals:
	void getLF(std::vector<cv::Mat> lf);
};
// class Worker : public QObject {
// 	Q_OBJECT
//    public:
// 	explicit Worker(QObject *parent = nullptr)
// 		: QObject(parent), core(std::make_unique<Core>()) {}
// 	~Worker() {}

// 	template <typename Func, typename... Args>

// 	void invoke(Func &&func, Args &&...args) {
// 		QMetaObject::invokeMethod(
// 			this, // 调用 Worker 的匿名槽函数
// 			[this, func = std::forward<Func>(func),
// 			 args = std::tuple(std::forward<Args>(args)...)]() mutable {
// 				// 使用 core.get() 绑定 Core 的 this 指针
// 				std::apply(
// 					[&](auto &&...params) {
// 						(core.get()->*func)(
// 							std::forward<decltype(params)>(params)...);
// 					},
// 					std::move(args));
// 			},
// 			Qt::QueuedConnection);
// 	}
// 	// void invoke(Func &&func, Args &&...args) {
// 	// 	QMetaObject::invokeMethod(
// 	// 		this,
// 	// 		[=]() mutable { (core.get()->*func)(std::forward<Args>(args)...); },
// 	// 		Qt::QueuedConnection);
// 	// }

//    private:
// 	std::unique_ptr<Core> core;
// };

} // namespace LFLoader

#endif