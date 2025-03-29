#ifndef UI_MAINWINDOW_H
#define UI_MAINWINDOW_H

#include <QDoubleSpinBox>
#include <QFileDialog>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QMainWindow>
#include <QObject>
#include <QPushButton>
#include <QSlider>
#include <QVBoxLayout>
#include <QWidget>

namespace Ui {
class MainWindow : public QMainWindow {
	Q_OBJECT
   public:
	void setupUi(QMainWindow* mainWindow);

   private:
	QGroupBox* setupModeGroup();
	QGroupBox* setupViewsGroup();
	QGroupBox* setupRefocusGroup();
	QGroupBox* setupSRGroup();
	QGroupBox* setupDEGroup();
};
}; // namespace Ui

#endif // UI_H
