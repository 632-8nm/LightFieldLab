#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include "lfdata.h"
#include "lfprocessor.h"
#include "ui.h"

class MainWindow : public QMainWindow {
	Q_OBJECT
public:
	MainWindow(QWidget* parent = nullptr);
	~MainWindow();

signals:
	void requestUpdateSAI(const cv::Mat& image);

public slots:
	void updateSAI(const cv::Mat& cvImg);
	void onLensletBrowseBtn();
	void viewValueUpdated(const LightFieldPtr& ptr);
	void onViewVerticalSliderUpdated(int value);
	void onViewHorizontalSliderUpdated(int value);
	void onRefocusAlphaChanged(int value);
	void onRefocusCropChanged(int value);

private:
	Ui::MainWindow* ui;
	LFProcessor* lfp;
	QThread* lfp_thread;
};
#endif // MAINWINDOW_H
