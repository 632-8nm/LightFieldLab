#include <QApplication>
#include <QMainWindow>
#include <cstring>
#include <opencv2/opencv.hpp>

#include "interface.h"
#include "lfloader.h"

int main(int argc, char* argv[]) {
	QApplication app(argc, argv);

	LFLoader::Worker			  lfloader;
	WindowBase<LFLoader::Worker>* windowBase =
		new WindowBase<LFLoader::Worker>(&lfloader);
	std::string path(argv[1]);
	bool		isRGB = strcmp(argv[2], "1") == 0 ? true : false;
	QObject::connect(
		windowBase->_button, &QPushButton::clicked, windowBase, [=]() {
			windowBase->_worker->invoke(&LFLoader::Core::loadSAI, path, isRGB);
		});
	// cv::Mat center = lfloader.invoke(&LFLoader::Core::getLF);

	windowBase->show();
	app.exec();
}