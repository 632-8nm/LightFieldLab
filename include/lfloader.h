#ifndef LFLOADER_H
#define LFLOADER_H

#include <QtCore/qobject.h>
#include <QtCore/qstring.h>
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
	explicit Core() : lf() {}
	void	   load(const std::string& path, const bool& isRGB);
	LightField lf;
};
class Worker : public QObject {
	Q_OBJECT
   public:
	explicit Worker(QObject* parent = nullptr);

   public slots:
	void printThreadId();
	void load(const QString& path, const bool& isRGB);
	void getLF() { emit lfUpdated(std::make_shared<LightField>(_core->lf)); }

   private:
	std::unique_ptr<Core> _core;
   signals:
	void lfUpdated(const LightFieldPtr& src);
};
class Worker_template : public WorkerBase<Core> {
	Q_OBJECT
   public:
	using WorkerBase<Core>::WorkerBase;
};

} // namespace LFLoader

#endif