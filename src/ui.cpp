#include "ui.h"

#include <QDoubleSpinBox>
#include <QFileDialog>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QLabel>
#include <QPushButton>
#include <QRadioButton>
#include <QSlider>
#include <QSpinBox>
#include <QStackedLayout>
#include <QStackedWidget>
#include <QVBoxLayout>
#include <QWidget>

void Ui::MainWindow::setupUi(QMainWindow *mainWindow) {
	QWidget *centralWidget = new QWidget(this);
	setCentralWidget(centralWidget);
	centralWidget->setMinimumSize(1600, 900);
}