#ifndef UI_H
#define UI_H

#include <QtWidgets/qcombobox.h>
#include <QtWidgets/qpushbutton.h>

#include <QComboBox>
#include <QDoubleSpinBox>
#include <QFileDialog>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QLabel>
#include <QMainWindow>
#include <QObject>
#include <QPushButton>
#include <QRadioButton>
#include <QSlider>
#include <QSpinBox>
#include <QVBoxLayout>
#include <QWidget>

namespace Ui {
class MainWindow : public QMainWindow {
	Q_OBJECT
public:
	void setupUi(QMainWindow* mainWindow);

	// 1
	QSlider *captureSlider, *colorSlider, *gpuSlider;

	QRadioButton* staticMode;
	QRadioButton* dynamicMode;
	QRadioButton* grayMode;
	QRadioButton* rgbMode;
	QPushButton*  lensletBrowseBtn;
	QPushButton*  whiteBrowseBtn;
	QLineEdit*	  lensletPathEdit;
	QLineEdit*	  whitePathEdit;

	// views
	QSlider*  verticalSlider;
	QSlider*  horizontalSlider;
	QSpinBox* verticalSpinBox;
	QSpinBox* horizontalSpinBox;

	// refocus
	QSlider*		cropSlider;
	QSpinBox*		cropSpinBox;
	QSlider*		alphaSlider;
	QDoubleSpinBox* alphaSpinBox;

	// super_resolution
	QComboBox*	 typeComboBox;
	QComboBox*	 scaleComboBox;
	QPushButton* SRButton;

	QLabel* rightPanel;

private:
	QGroupBox* setupModeGroup();
	QGroupBox* setupViewsGroup();
	QGroupBox* setupRefocusGroup();
	QGroupBox* setupSRGroup();
	QGroupBox* setupDEGroup();
};
}; // namespace Ui

#endif // UI_H
