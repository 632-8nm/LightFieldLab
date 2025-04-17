#ifndef LFPROCESSOR_H
#define LFPROCESSOR_H

#include <QtCore/qstring.h>
#include <QtCore/qthread.h>
#include <QtCore/qtmetamacros.h>

#include <QObject>
#include <QThread>

#include "lfdata.h"
#include "lfloader.h"
#include "lfrefocuser.h"

class LFProcessor : public QObject {
	Q_OBJECT
   public:
	enum WorkerType {
		LOADER = 0,
		REFOCUSER,
		SUPER_RESOLUTION, // 示例：可扩展其他类型
		DEPTH_ESTIMATION,
		WORKER_COUNT // 自动计数
	};
	explicit LFProcessor(QObject* parent = nullptr);
	~LFProcessor();

	LFLoader::Worker* loader() const {
		return dynamic_cast<LFLoader::Worker*>(workers[LOADER]);
	}
	LFRefocus::Worker* refocuser() const {
		return dynamic_cast<LFRefocus::Worker*>(workers[REFOCUSER]);
	}

   public slots:
	void printThreadId() {
		std::cout << "LFProcessor threadId: " << QThread::currentThreadId()
				  << " = printThreadId called!" << std::endl;
	}
	void onLFUpdated(const LightFieldPtr& src);
	void onGpuSliderValueChanged(int value);

   signals:
	void updateSAI(const cv::Mat& image);
	void updateLF(const LightFieldPtr& ptr);

   public:
	void initWorker(WorkerType type);
	void toFLoat(LightFieldPtr ptr);

	cv::Mat		  sai;
	QThread*	  threads[WORKER_COUNT];
	QObject*	  workers[WORKER_COUNT]; // 基类指针存储所有worker
	LightFieldPtr lf, lf_float;
	QString		  lensletImagePath, whiteImagePath;
	bool		  isRgb = false, isGpu = false;
	int			  sai_row = 8, sai_col = 8, crop = 0;
	float		  alpha = 1.0;
};

#endif