#ifndef REFOCUS_H
#define REFOCUS_H
#include <QtCore/qtmetamacros.h>

#include <QObject>
#include <QTHread>
#include <memory>
#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <vector>
class Refocus : public QObject {
	Q_OBJECT
   public:
	explicit Refocus(QObject* parent = nullptr);
	explicit Refocus(const std::vector<cv::Mat>& src = std::vector<cv::Mat>(),
					 QObject*					 parent = nullptr);
	~Refocus();

	void para_init(const std::vector<cv::Mat>& src);
	void refocus(float alpha, int offset);
	void setLF(const std::vector<cv::Mat>& src);
	void setGPU(bool isGPU);

	bool	getGPU() const { return _isGPU; }
	cv::Mat getRefocusedImage() const { return _refocusedImage; }
   public slots:
	void work_test();
   signals:
	void refocusFinished(std::chrono::duration<double> elapsed);

   private:
	std::vector<cv::Mat>				   _lf;
	std::unique_ptr<std::vector<cv::UMat>> _lf_gpu;
	bool								   _isGPU = false;
	int									   _views, _len, _center, _type;
	cv::Mat								   _xmap, _ymap, _refocusedImage;
	cv::UMat							   _xmap_gpu, _ymap_gpu;
	cv::Size							   _size;
};
#endif