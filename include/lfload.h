#ifndef LFLOAD_H
#define LFLOAD_H

#include <QObject>
#include <QString>
#include <QThread>
#include <QWidget>
#include <opencv2/opencv.hpp>

#include "lfdata.h"
class LFLoad : public QObject {
	Q_OBJECT
public:
	explicit LFLoad(QObject* parent = nullptr);

	// LightField lf;
public slots:
	void printThreadId();
	void load(const QString& path_, bool isRGB);
	// void getLF() { emit finished(std::make_shared<LightField>(lf)); }

signals:
	void finished(const LightFieldPtr& src);
};

#endif