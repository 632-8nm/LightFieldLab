#include "refocus.h"

#include <opencv2/core/hal/interface.h>

#include <cmath>
#include <opencv2/core/types.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>

Refocus::Refocus(QObject* parent, int views, int height, int width)
	: QObject(parent), views_(views), height_(height), width_(width) {
	// Constructor implementation
	len_ = std::sqrt(views);

	xgrid_ = cv::Mat(height_, 1, CV_32F);
	ygrid_ = cv::Mat(1, width_, CV_32F);
	for (int x = 0; x < height_; x++) {
		xgrid_.at<float>(x, 0) = static_cast<float>(x);
	}
	for (int y = 0; y < width_; y++) {
		ygrid_.at<float>(0, y) = static_cast<float>(y);
	}
	xgrid_ = cv::repeat(xgrid_, 1, width_);
	ygrid_ = cv::repeat(ygrid_, height_, 1);
}
Refocus::~Refocus() {
	// Destructor implementation
}
cv::Mat Refocus::refocus(const std::vector<cv::Mat>& input, float alpha,
						 int offset) {
	cv::Mat output(input[0].size(), CV_32F, cv::Scalar(0));
	float	factor = 1.0f / (1 - 1 / alpha);
	for (int i = 0; i < input.size(); i++) {
		int		row = i / len_;
		int		col = i % len_;
		cv::Mat temp;
		cv::remap(input[i], temp, ygrid_ + factor * (row - center_),
				  xgrid_ + factor * (col - center_), cv::INTER_LINEAR,
				  cv::BORDER_CONSTANT, 0);
		cv::Mat temp2;
		cv::add(temp, output, output);
	}
	cv::divide((len_ - offset) * (len_ - offset), output, output);
	output.convertTo(output, input[0].type());

	return output;
}