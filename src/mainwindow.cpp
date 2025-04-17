#include "mainwindow.h"

#include <QtCore/qlogging.h>
#include <QtCore/qnamespace.h>
#include <QtCore/qtmetamacros.h>
#include <QtGui/qimage.h>
#include <QtWidgets/qslider.h>

#include <QDoubleSpinBox>
#include <QFileDialog>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QImage>
#include <QLineEdit>
#include <QMetaMethod>
#include <QMetaObject>
#include <QPixmap>
#include <QPushButton>
#include <QSlider>
#include <QString>
#include <QThread>
#include <QVBoxLayout>
#include <iostream>
#include <opencv2/core/mat.hpp>
#include <ostream>

#include "lfdata.h"
#include "lfloader.h"
#include "lfprocessor.h"
#include "lfrefocuser.h"
#include "ui.h"

MainWindow::MainWindow(QWidget *parent) : QMainWindow(parent) {
	std::cout << "MainWindow threadId: " << QThread::currentThreadId()
			  << std::endl;

	ui		   = new Ui::MainWindow();
	lfp		   = new LFProcessor();
	lfp_thread = new QThread();

	ui->setupUi(this);

	lfp->moveToThread(lfp_thread);
	connect(lfp_thread, &QThread::started, lfp, &LFProcessor::printThreadId);
	lfp_thread->start();

	// connect(
	// 	ui->gpuSlider, &QSlider::valueChanged, lfp,
	// 	[&](int value) {
	// 		lfp->isGpu = static_cast<bool>(value);
	// 		lfp->refocuser()->setGpu(lfp->isGpu);
	// 		// lfp->lf->toGpu();
	// 	},
	// 	Qt::QueuedConnection);
	connect(ui->gpuSlider, &QSlider::valueChanged, lfp,
			&LFProcessor::onGpuSliderValueChanged, Qt::QueuedConnection);

	// 0. load
	connect(ui->lensletBrowseBtn, &QPushButton::clicked, this,
			&MainWindow::onLensletBrowseBtn);
	connect(lfp->loader(), &LFLoader::Worker::lfUpdated, this,
			&MainWindow::viewValueUpdated);

	// 1. view
	connect(ui->verticalSlider, &QSlider::valueChanged, this,
			&MainWindow::onViewVerticalSliderUpdated); // -> requestUpdateSAI
	connect(ui->horizontalSlider, &QSlider::valueChanged, this,
			&MainWindow::onViewHorizontalSliderUpdated); // -> requestUpdateSAI
	connect(this, &MainWindow::requestUpdateSAI, this, &MainWindow::updateSAI);
	connect(lfp, &LFProcessor::updateSAI, this, &MainWindow::updateSAI);

	// 2. refocus

	connect(ui->alphaSlider, &QSlider::valueChanged, this,
			&MainWindow::onRefocusAlphaChanged);
	connect(ui->cropSlider, &QSlider::valueChanged, this,
			&MainWindow::onRefocusCropChanged);
	connect(lfp->refocuser(), &LFRefocus::Worker::requestUpdateSAI, this,
			&MainWindow::updateSAI);
}

MainWindow::~MainWindow() {
	if (lfp_thread && lfp_thread->isRunning()) {
		lfp_thread->quit();
		lfp_thread->wait();
	}
	delete lfp;
	delete lfp_thread;
	delete ui;
}
void MainWindow::onLensletBrowseBtn() {
	if (!ui->lensletPathEdit->text().isEmpty()) {
		lfp->lensletImagePath = this->ui->lensletPathEdit->text();
		lfp->isRgb			  = this->ui->colorSlider->value() ? true : false;
		qDebug() << lfp->lensletImagePath;
		QMetaObject::invokeMethod(
			lfp->workers[LFProcessor::LOADER], "load", Qt::QueuedConnection,
			Q_ARG(QString, lfp->lensletImagePath), Q_ARG(bool, lfp->isRgb));
	}
}
void MainWindow::updateSAI(const cv::Mat &cvImg) {
	// std::cout << "MainWindow threadId: " << QThread::currentThreadId()
	// 		  << std::endl;
	QImage qImg;
	if (cvImg.channels() == 1) {
		qImg = QImage(cvImg.data, cvImg.cols, cvImg.rows, cvImg.step,
					  QImage::Format_Grayscale8);
	} else {
		qImg = QImage(cvImg.data, cvImg.cols, cvImg.rows, cvImg.step,
					  QImage::Format_BGR888);
	}
	ui->rightPanel->setPixmap(QPixmap::fromImage(qImg));
}
void MainWindow::viewValueUpdated(const LightFieldPtr &ptr) {
	int rows = ptr->rows, cols = ptr->cols;
	ui->verticalSlider->setRange(1, rows);
	ui->verticalSpinBox->setRange(1, rows);
	ui->horizontalSlider->setRange(1, cols);
	ui->horizontalSpinBox->setRange(1, cols);
	ui->verticalSlider->setValue(lfp->sai_row);
	ui->verticalSpinBox->setValue(lfp->sai_row);
	ui->horizontalSlider->setValue(lfp->sai_col);
	ui->horizontalSpinBox->setValue(lfp->sai_col);
	ui->cropSlider->setRange(0, rows / 2);
	ui->cropSpinBox->setRange(0, rows / 2);
}
void MainWindow::onViewVerticalSliderUpdated(int value) {
	if (lfp->lf == nullptr) {
		return;
	}
	lfp->sai_row = value;
	emit requestUpdateSAI(lfp->lf->getSAI(lfp->sai_row - 1, lfp->sai_col - 1));
}
void MainWindow::onViewHorizontalSliderUpdated(int value) {
	if (lfp->lf == nullptr) {
		return;
	}
	lfp->sai_col = value;
	emit requestUpdateSAI(lfp->lf->getSAI(lfp->sai_row - 1, lfp->sai_col - 1));
}
void MainWindow::onRefocusAlphaChanged(int value) {
	if (lfp->lf == nullptr) {
		return;
	}
	lfp->alpha = value / 100.0f;
	QMetaObject::invokeMethod(lfp->workers[LFProcessor::REFOCUSER], "refocus",
							  Qt::QueuedConnection, Q_ARG(float, lfp->alpha),
							  Q_ARG(int, lfp->crop));
}
void MainWindow::onRefocusCropChanged(int value) {
	if (lfp->lf == nullptr) {
		return;
	}
	lfp->crop = value;
	QMetaObject::invokeMethod(lfp->workers[LFProcessor::REFOCUSER], "refocus",
							  Qt::QueuedConnection, Q_ARG(float, lfp->alpha),
							  Q_ARG(int, lfp->crop));
}
