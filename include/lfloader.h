#ifndef LOADER_H
#define LOADER_H

#include <QtWidgets/qwidget.h>

#include <QObject>
#include <QWidget>
#include <opencv2/core/mat.hpp>
#include <string>
#include <vector>

class LFLoader : public QWidget {
	Q_OBJECT
   public:
	explicit LFLoader(QWidget *parent = nullptr);
	~LFLoader();

	void				 loadSAI(std::string path, bool isRGB);
	std::vector<cv::Mat> getLF() const { return LF; }
	std::vector<cv::Mat> getLF_float32() const { return LF_float32; }

	//    private:
	std::vector<cv::Mat> LF;
	std::vector<cv::Mat> LF_float32;
};
#endif