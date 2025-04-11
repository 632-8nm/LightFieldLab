#ifndef LFLOADER_H
#define LFLOADER_H

#include <QtCore/qobject.h>
#include <QtCore/qtmetamacros.h>

#include <QObject>
#include <QThread>
#include <QWidget>
#include <memory>
#include <opencv2/core/mat.hpp>
#include <opencv2/highgui.hpp>
#include <string>

#include "lfdata.h"
#include "worker_base.h"

namespace LFLoader {
class Core {
   public:
	explicit Core() {
		_lf			= LightField();
		_lf_float32 = LightField();
	}
	~Core() {}

	void load(const std::string& path, const bool& isRGB);

	LightField getLF() const { return _lf; }
	LightField getLF_float32() const { return _lf_float32; }

   private:
	LightField _lf;
	LightField _lf_float32;
};
class Worker : public QObject {
	Q_OBJECT
   public:
	explicit Worker(QObject* parent = nullptr) : QObject(parent) {
		_core = std::make_unique<Core>();
	}
	~Worker() {}

	void load(const std::string& path, const bool& isRGB) {
		std::cout << "load thread: " << QThread::currentThreadId() << std::endl;
		_core->load(path, isRGB);
	};
	void getLF() {
		std::cout << "getLF thread: " << QThread::currentThreadId()
				  << std::endl;
		emit sendLF(_core->getLF());
	}
	std::unique_ptr<Core> _core;
   signals:
	void sendLF(const LightField& lf);

	//    private:
};
class Worker_template : public WorkerBase<Core> {
	Q_OBJECT
   public:
	using WorkerBase<Core>::WorkerBase;
};

} // namespace LFLoader

#endif