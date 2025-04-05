#include "Controller.h"

#include <QtCore/qobject.h>

#include <memory>
#include <opencv2/core/mat.hpp>
#include <vector>

#include "refocus.h"

Controller::Controller(QObject* parent) : QObject(parent) {
	// std::vector<cv::Mat> src;
	// refocuser = std::make_unique<Refocus>(src, this);
}
Controller::~Controller() {}