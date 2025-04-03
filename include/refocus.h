#ifndef REFOCUS_H
#define REFOCUS_H
#include <QObject>
#include <QTHread>
#include <opencv2/core.hpp>
#include <vector>
class Refocus : public QObject {
	Q_OBJECT
   public:
	explicit Refocus(QObject* parent = nullptr, int views = 0, int height = 0,
					 int width = 0);
	~Refocus();

	cv::Mat refocus(const std::vector<cv::Mat>& input, float alpha, int offset);

   private:
	int		views_, height_, width_, len_, center_;
	cv::Mat xgrid_, ygrid_;
};
#endif