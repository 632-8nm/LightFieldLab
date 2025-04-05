#ifndef REFOCUS_H
#define REFOCUS_H
#include <QObject>
#include <QTHread>
#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <vector>
class Refocus : public QObject {
	Q_OBJECT
   public:
	explicit Refocus(QObject*					 parent = nullptr,
					 const std::vector<cv::Mat>& input	= {});
	~Refocus();

	cv::Mat refocus(const std::vector<cv::Mat>& input, float alpha, int offset);

   private:
	int		 views_, len_, center_, type_;
	cv::Mat	 xgrid_, ygrid_;
	cv::Size size_;
};
#endif