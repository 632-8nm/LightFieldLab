#ifndef LFDATA_H
#define LFDATA_H

#include <cmath>
#include <opencv2/opencv.hpp>
#include <vector>

class LightField {
   public:
	LightField() {}
	LightField(const std::vector<cv::Mat>& lf) : lf_(lf) {
		validate(lf);
		computeDimensions(lf);
	}

	LightField(std::vector<cv::Mat>&& lf) : lf_(std::move(lf)) {
		validate(lf);
		computeDimensions(lf);
	}
	std::vector<cv::Mat> get() const { return lf_; }
	cv::Mat				 getCenter() const { return lf_[size / 2]; }
	bool				 empty() const { return lf_.empty(); }
	void				 clear() { lf_.clear(); }

   private:
	void validate(const std::vector<cv::Mat>& lf) {
		if (lf.empty()) {
			throw std::invalid_argument(
				"LightField requires a non-empty vector");
		}
		int size = lf.size();
		int rows = static_cast<int>(std::sqrt(size));
		if (rows * rows != size) {
			throw std::invalid_argument(
				"LightField size must be a perfect square");
		}
		if (lf[0].empty()) {
			throw std::invalid_argument("Input images must not be empty");
		}
	}
	void computeDimensions(const std::vector<cv::Mat>& lf) {
		size	 = lf.size();
		rows	 = static_cast<int>(std::sqrt(lf.size()));
		cols	 = rows;
		height	 = lf[0].rows;
		width	 = lf[0].cols;
		channels = lf[0].channels();
	}

   public:
	int size = 0, rows = 0, cols = 0, height = 0, width = 0, channels = 0;
	std::vector<cv::Mat> lf_{};
	// std::vector<cv::Mat> lf_ = std::vector<cv::Mat>(0);
};

#endif