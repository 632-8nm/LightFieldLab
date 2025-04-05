#include "refocus.h"

#include <opencv2/core/hal/interface.h>

#include <cmath>
#include <iostream>
#include <opencv2/core/base.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>

Refocus::Refocus(QObject* parent, const std::vector<cv::Mat>& input)
	: QObject(parent) {
	// Constructor implementation
	views_ = input.size();
	len_   = static_cast<int>(std::sqrt(views_));
	size_  = input[0].size();
	type_  = input[0].type();

	xgrid_ = cv::Mat(size_.height, 1, CV_32FC1);
	ygrid_ = cv::Mat(1, size_.width, CV_32FC1);
	for (int x = 0; x < size_.height; x++) {
		xgrid_.at<float>(x, 0) = static_cast<float>(x);
	}
	for (int y = 0; y < size_.width; y++) {
		ygrid_.at<float>(0, y) = static_cast<float>(y);
	}
	xgrid_ = cv::repeat(xgrid_, 1, size_.width);
	ygrid_ = cv::repeat(ygrid_, size_.height, 1);
}
Refocus::~Refocus() {
	// Destructor implementation
}
cv::Mat Refocus::refocus(const std::vector<cv::Mat>& input, float alpha,
						 int offset) {
	cv::Mat interp_output;
	cv::Mat output(size_, type_, cv::Scalar(0));
	float	factor = 1.0f - 1.0f / alpha;
	for (int i = 0; i < input.size(); i++) {
		int row = i / len_;
		int col = i % len_;
		if (row < offset || col < offset || row >= len_ - offset
			|| col >= len_ - offset) {
			continue;
		}

		cv::remap(input[i], interp_output, ygrid_ + factor * (col - center_),
				  xgrid_ + factor * (row - center_), cv::INTER_LINEAR,
				  cv::BORDER_REPLICATE);

		cv::add(interp_output, output, output);
	}
	cv::divide(output, (len_ - 2 * offset) * (len_ - 2 * offset), output);

	return output;
}