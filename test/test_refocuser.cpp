#include <QtCore/qstring.h>
#include <QtCore/qthread.h>
#include <QtCore/qtimer.h>
#include <QtCore/qtmetamacros.h>
#include <QtWidgets/qapplication.h>
#include <QtWidgets/qboxlayout.h>
#include <QtWidgets/qlabel.h>
#include <QtWidgets/qpushbutton.h>
#include <opencv2/core/hal/interface.h>

#include <QApplication>
#include <QLabel>
#include <QMainWindow>
#include <QTimer>
#include <QVBoxLayout>
#include <cstring>
#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <vector>

#include "lfloader.h"
#include "lfrefocuser.h"
class Window : public QMainWindow {
	// Q_OBJECT
   public:
	Window(int argc, char* argv[], QWidget* parent = nullptr)
		: QMainWindow(parent), counter(0) {
		std::string path  = std::string(argv[1]);
		bool		isRGB = strcmp(argv[2], "1") == 0 ? true : false;
		LFLoader	loader;
		loader.loadSAI(path, isRGB);
		// std::vector<cv::Mat> LF			= loader.getLF();
		std::vector<cv::Mat> LF_float32 = loader.getLF_float32();

		bool		  isGpu		= false;
		QLFRefocuser* refocuser = new QLFRefocuser(LF_float32, this);
		refocuser->setGpuRequest(false);
		// QThread* thread = new QThread(this);
		// refocuser->moveToThread(thread);

		QWidget* centralWidget = new QWidget(this);
		setCentralWidget(centralWidget);

		QVBoxLayout* layout		   = new QVBoxLayout(centralWidget);
		QLabel*		 counterLabel  = new QLabel("Counter: 0", this);
		QLabel*		 refocusLabel  = new QLabel("Refocus time: 0", this);
		QPushButton* refocusButton = new QPushButton("Refocus", this);
		QHBoxLayout* hLayout	   = new QHBoxLayout(this);
		QLabel*		 gpuLable	   = new QLabel("Current use: CPU", this);
		QPushButton* gpuButton	   = new QPushButton("Switch", this);
		layout->addWidget(counterLabel);
		layout->addWidget(refocusButton);
		layout->addWidget(refocusLabel);
		hLayout->addWidget(gpuLable);
		hLayout->addWidget(gpuButton);
		layout->addLayout(hLayout);

		connect(refocusButton, &QPushButton::clicked, this,
				[refocuser, refocusButton]() {
					refocuser->refocusRequest(1.5, 2);
				});
		connect(refocuser, &QLFRefocuser::refocusFinished, this,
				[refocusLabel](std::chrono::duration<double> elapsed) {
					refocusLabel->setText(
						QString("Refocus time: %1").arg(elapsed.count()));
				});
		connect(gpuButton, &QPushButton::clicked, this,
				[&isGpu, gpuLable, refocuser]() {
					isGpu = !isGpu;
					refocuser->setGpuRequest(isGpu);
					gpuLable->setText(
						QString("Current use: %1").arg(isGpu ? "GPU" : "CPU"));
				});

		QTimer* timer = new QTimer(this);
		timer->start(200); // 每0.2秒更新一次
		connect(timer, &QTimer::timeout, this, [this, counterLabel]() {
			counter += 0.2;
			counterLabel->setText(QString("Counter: %1").arg(counter));
		});
	}

   private:
	float counter = 0; // 计数器变量
};
int main(int argc, char* argv[]) {
	std::cout << "Testing Refocus class" << std::endl;

	QApplication app(argc, argv);
	Window		 window(argc, argv, nullptr);
	window.show();
	return app.exec();
}