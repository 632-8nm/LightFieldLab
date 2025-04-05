#include <QtCore/qstring.h>
#include <QtCore/qthread.h>
#include <QtCore/qtimer.h>
#include <QtCore/qtmetamacros.h>
#include <QtWidgets/qapplication.h>
#include <QtWidgets/qlabel.h>
#include <opencv2/core/hal/interface.h>

#include <QApplication>
#include <QLabel>
#include <QMainWindow>
#include <QTimer>
#include <QVBoxLayout>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/opencv.hpp>
#include <vector>

#include "refocus.h"
class Window : public QMainWindow {
	// Q_OBJECT
   public:
	Window(QWidget* parent = nullptr, Refocus* refocuser = nullptr)
		: QMainWindow(parent), counter(0) {
		QWidget* centralWidget = new QWidget(this);
		setCentralWidget(centralWidget);

		QVBoxLayout* layout		  = new QVBoxLayout(centralWidget);
		QLabel*		 counterLabel = new QLabel("Counter: 0", this);
		QLabel*		 refocusLabel = new QLabel("Refocus time: 0", this);
		layout->addWidget(counterLabel);
		layout->addWidget(refocusLabel);

		QTimer* timer = new QTimer(this);
		timer->start(1000); // 每秒更新一次
		connect(timer, &QTimer::timeout, this, [this, counterLabel]() {
			counter++;
			counterLabel->setText(QString("Counter: %1").arg(counter));
		});

		QThread* thread = new QThread(this);
		refocuser->moveToThread(thread);
		refocuser->setGPU(true);
		connect(thread, &QThread::started, refocuser, &Refocus::work_test);
		connect(
			refocuser, &Refocus::refocusFinished, this,
			[refocuser, refocusLabel](std::chrono::duration<double> elapsed) {
				refocusLabel->setText(
					QString("Refocus time: %1").arg(elapsed.count()));
			});
		thread->start();
	}

   private:
	int counter; // 计数器变量
};
int main(int argc, char* argv[]) {
	std::cout << "Testing Refocus class" << std::endl;

	std::string				 path = std::string(argv[1]);
	std::vector<std::string> filenames;

	for (const auto& entry : std::filesystem::directory_iterator(path)) {
		if (entry.is_regular_file()) {
			filenames.push_back(
				entry.path().filename().string()); // 只保存文件名
		}
	}

	// 排序
	std::sort(filenames.begin(), filenames.end());

	std::vector<cv::Mat> LF, LF_float32;
	// 输出
	for (const auto& name : filenames) {
		// std::cout << name << std::endl;
		cv::Mat img, img_float32;
		if (strcmp(argv[2], "gray") == 0) {
			img = cv::imread(path + name, cv::IMREAD_GRAYSCALE);
		} else {
			img = cv::imread(path + name, cv::IMREAD_COLOR);
		}

		img.convertTo(img_float32, CV_32FC(img.channels()));
		LF.push_back(img);
		LF_float32.push_back(img_float32);
	}

	// cv::imshow("center", LF[112]);
	// cv::waitKey(0);

	Refocus* refocuser = new Refocus(LF_float32, nullptr);
	float	 alpha	   = 1.5;
	int		 offset	   = 2;
	// cpu计算
	// rfc->setGPU(false);
	// auto start = std::chrono::high_resolution_clock::now();
	// rfc->refocus(alpha, offset);
	// auto end = std::chrono::high_resolution_clock::now();
	// std::chrono::duration<double> elapsed = end - start;
	// std::cout << "====== Refocus time: " << elapsed.count()
	// 		  << " seconds ======" << std::endl;
	// cv::Mat refocusedImg = rfc->getRefocusedImage();
	// refocusedImg.convertTo(refocusedImg, CV_8UC(refocusedImg.channels()));
	// cv::imshow("1 cpu", refocusedImg);
	// cv::waitKey(0);

	// gpu计算
	// rfc->setGPU(true);
	// start = std::chrono::high_resolution_clock::now();
	// rfc->refocus(alpha, offset);
	// end		= std::chrono::high_resolution_clock::now();
	// elapsed = end - start;
	// std::cout << "====== Refocus time: " << elapsed.count()
	// 		  << " seconds ======" << std::endl;
	// refocusedImg = rfc->getRefocusedImage();
	// refocusedImg.convertTo(refocusedImg, CV_8UC(refocusedImg.channels()));
	// cv::imshow("2 gpu", refocusedImg);
	//   cv::waitKey(0);

	// cpu计算
	// rfc->setGPU(false);
	// start = std::chrono::high_resolution_clock::now();
	// rfc->refocus(alpha, offset);
	// end		= std::chrono::high_resolution_clock::now();
	// elapsed = end - start;
	// std::cout << "====== Refocus time: " << elapsed.count()
	// 		  << " seconds ======" << std::endl;

	// cv::Mat refocusedImg = rfc->getRefocusedImage();
	// refocusedImg.convertTo(refocusedImg, CV_8UC(refocusedImg.channels()));
	// cv::imshow("3 cpu", refocusedImg);
	// cv::waitKey(0);

	QApplication app(argc, argv);
	Window		 window(nullptr, refocuser);
	window.show();
	return app.exec();

	delete refocuser;
	// return 0;
}