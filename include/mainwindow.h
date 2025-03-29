#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QGroupBox>
#include <QLabel>
#include <QMainWindow>
#include <QPushButton>
#include <QRadioButton>
#include <QStackedWidget>

QT_BEGIN_NAMESPACE
namespace Ui {
class MainWindow;
}
QT_END_NAMESPACE

class MainWindow : public QMainWindow {
	Q_OBJECT
   public:
	MainWindow(QWidget* parent = nullptr);
	~MainWindow();

   private:
	QGroupBox* setupModeGroup();
	QGroupBox* setupViewsGroup();
	QGroupBox* setupRefocusGroup();
	QGroupBox* setupSRGroup();
	QGroupBox* setupDEGroup();
};
#endif // MAINWINDOW_H
