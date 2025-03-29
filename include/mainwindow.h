#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QGroupBox>
#include <QLabel>
#include <QMainWindow>
#include <QPushButton>
#include <QRadioButton>
#include <QStackedWidget>

#include "ui.h"

class MainWindow : public QMainWindow {
	Q_OBJECT
   public:
	MainWindow(QWidget* parent = nullptr);
	~MainWindow();

   private:
	Ui::MainWindow* ui;
	QGroupBox*		setupModeGroup();
	QGroupBox*		setupViewsGroup();
	QGroupBox*		setupRefocusGroup();
	QGroupBox*		setupSRGroup();
	QGroupBox*		setupDEGroup();
};
#endif // MAINWINDOW_H
