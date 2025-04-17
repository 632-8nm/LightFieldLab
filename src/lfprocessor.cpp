#include "lfprocessor.h"

#include <QtCore/qnamespace.h>
#include <QtCore/qobject.h>
#include <QtCore/qstring.h>
#include <QtCore/qthread.h>
#include <QtCore/qtmetamacros.h>
#include <opencv2/core/hal/interface.h>

#include <QMetaMethod>
#include <QMetaObject>
#include <memory>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/matx.hpp>
#include <opencv2/core/persistence.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>

#include "lfdata.h"
#include "lfloader.h"
#include "lfrefocuser.h"

LFProcessor::LFProcessor(QObject* parent) : QObject(parent) {
	memset(threads, 0, sizeof(threads));
	memset(workers, 0, sizeof(workers));

	initWorker(LOADER);
	initWorker(REFOCUSER);

	connect(loader(), &LFLoader::Worker::lfUpdated, this,
			&LFProcessor::onLFUpdated, Qt::QueuedConnection);
	connect(this, &LFProcessor::updateLF, refocuser(),
			&LFRefocus::Worker::lfUpdated, Qt::QueuedConnection);
}
LFProcessor::~LFProcessor() {
	for (int i = 0; i < WORKER_COUNT; ++i) {
		if (threads[i]) {
			threads[i]->quit();
			threads[i]->wait();
			delete workers[i]; // 自动在正确线程删除
			delete threads[i];
		}
	}
}
void LFProcessor::initWorker(WorkerType type) {
	threads[type] = new QThread(this);

	switch (type) {
		case LOADER:
			workers[type] = new LFLoader::Worker();
			break;
		case REFOCUSER:
			workers[type] = new LFRefocus::Worker();
			break;
		// 扩展点：新增类型在此添加case
		default:
			qFatal("Unknown worker type");
	}

	workers[type]->moveToThread(threads[type]);

	// 统一连接线程启动打印
	connect(threads[type], &QThread::started, workers[type], [type, this]() {
		qDebug() << "Worker" << type
				 << "running in thread:" << QThread::currentThreadId();
	});

	threads[type]->start();
}
void LFProcessor::onLFUpdated(const LightFieldPtr& ptr) {
	if (lf && lf_float) {
		lf->clear();
		lf_float->clear();
	}

	lf		 = ptr;
	lf_float = std::make_shared<LightField>(*ptr);
	lf_float->toFloat();
	lf_float->toGpu();

	sai_row = (1 + ptr->rows) / 2;
	sai_col = (1 + ptr->cols) / 2;
	sai		= lf->getSAI(sai_row, sai_col);
	emit updateSAI(sai);
	emit updateLF(lf_float);
}

void LFProcessor::onGpuSliderValueChanged(int value) {
	isGpu = static_cast<bool>(value);
	refocuser()->setGpu(isGpu);
}