#include <QtWidgets/qapplication.h>

#include <cstring>
#include <opencv2/highgui.hpp>

#include "lfloader.h"

int main(int argc, char* argv[]) {
	QApplication app(argc, argv);

	LFLoader	loader;
	std::string path(argv[1]);
	bool		isRGB = strcmp(argv[2], "1") == 0 ? true : false;
	loader.loadSAI(path, isRGB);
	std::vector<cv::Mat> LF = loader.getLF();

	cv::imshow("center", LF[112]);
	cv::waitKey(0);

	return 0;
}