#include "ui.h"

#include <QtWidgets/qboxlayout.h>
#include <QtWidgets/qlabel.h>

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

namespace Ui {
void MainWindow::setupUi(QMainWindow *mainWindow) {
	QWidget *centralWidget = new QWidget(mainWindow);
	mainWindow->setCentralWidget(centralWidget);
	centralWidget->setMinimumSize(1280, 720);
	QHBoxLayout *mainLayout = new QHBoxLayout(centralWidget);

	// // === 左侧功能区 ===
	QWidget		*leftPanel	= new QWidget();
	QVBoxLayout *leftLayout = new QVBoxLayout(leftPanel);

	QGroupBox *modeGroup	= setupModeGroup();	   // 1. 模式切换组
	QGroupBox *viewsGroup	= setupViewsGroup();   // 2. 视角变换组
	QGroupBox *refocusGroup = setupRefocusGroup(); // 3. 重聚焦组
	QGroupBox *srGroup		= setupSRGroup();	   // 4. 超分辨组
	QGroupBox *deGroup		= setupDEGroup();	   // 5. 深度估计组

	leftLayout->addWidget(modeGroup);
	leftLayout->addWidget(viewsGroup);
	leftLayout->addWidget(refocusGroup);
	leftLayout->addWidget(srGroup);
	leftLayout->addWidget(deGroup);
	leftLayout->addStretch(); // 占位填充
	leftLayout->addStretch();

	// === 右侧显示区 ===
	QLabel *rightPanel = new QLabel("图像显示区域");
	rightPanel->setAlignment(Qt::AlignCenter);
	rightPanel->setStyleSheet("border: 1px solid gray;");

	// 主布局组装
	mainLayout->addWidget(leftPanel, 1);
	mainLayout->addWidget(rightPanel, 3);
}
// 1. 数据模式组
QGroupBox *MainWindow::setupModeGroup() {
	QGroupBox	 *modeGroup	  = new QGroupBox("数据模式");
	QRadioButton *staticMode  = new QRadioButton("静态数据");
	QRadioButton *dynamicMode = new QRadioButton("动态数据流");
	QVBoxLayout	 *modeLayout  = new QVBoxLayout();
	QHBoxLayout	 *checkLayout = new QHBoxLayout();
	checkLayout->addWidget(staticMode);
	checkLayout->addWidget(dynamicMode);
	modeLayout->addLayout(checkLayout);
	modeGroup->setLayout(modeLayout);
	staticMode->setChecked(true); // 默认静态模式

	// 输入区域堆叠布局
	QStackedWidget *inputStack = new QStackedWidget(modeGroup);

	// --- 静态模式页面 ---
	QWidget		*staticPage		  = new QWidget();
	QVBoxLayout *staticLayout	  = new QVBoxLayout(staticPage);
	QPushButton *btnWhiteImage	  = new QPushButton("加载白图像");
	QPushButton *btnLensletImage  = new QPushButton("加载微透镜图像");
	QLabel		*labelWhitePath	  = new QLabel("未选择文件");
	QLabel		*labelLensletPath = new QLabel("未选择文件");
	staticLayout->addWidget(btnWhiteImage);
	staticLayout->addWidget(labelWhitePath);
	staticLayout->addWidget(btnLensletImage);
	staticLayout->addWidget(labelLensletPath);
	inputStack->addWidget(staticPage);

	// --- 动态模式页面 ---
	QWidget		*dynamicPage   = new QWidget();
	QHBoxLayout *dynamicLayout = new QHBoxLayout(dynamicPage);
	QPushButton *btnCapture	   = new QPushButton("采集");
	QPushButton *btnDecode	   = new QPushButton("解码");
	dynamicLayout->addWidget(btnCapture);
	dynamicLayout->addWidget(btnDecode);
	inputStack->addWidget(dynamicPage);

	modeLayout->addWidget(inputStack);

	connect(staticMode, &QRadioButton::toggled, this,
			[inputStack](bool isStatic) {
				inputStack->setCurrentIndex(isStatic ? 0 : 1);
			});
	connect(
		btnWhiteImage, &QPushButton::clicked, this, [this, labelWhitePath]() {
			QString path = QFileDialog::getOpenFileName(
				this, "选择白图像", "", "Images (*.png *.tiff *.bmp *.mim)");
			if (!path.isEmpty()) {
				labelWhitePath->setText(path);
				// TODO: 实际加载图像逻辑
			}
		});
	connect(btnLensletImage, &QPushButton::clicked, this,
			[this, labelLensletPath]() {
				QString path = QFileDialog::getOpenFileName(
					this, "选择微透镜图像", "",
					"Images (*.png *.tiff *.bmp *.mim)");
				if (!path.isEmpty()) {
					labelLensletPath->setText(path);
					// TODO: 实际加载图像逻辑
				}
			});
	connect(btnCapture, &QPushButton::clicked, this, []() {
		// TODO: 初始化摄像头或视频流
	});
	connect(btnDecode, &QPushButton::clicked, this, []() {
		// TODO: 启动动态数据解码
	});

	return modeGroup;
}

QGroupBox *MainWindow::setupViewsGroup() {
	QGroupBox	*viewGroup	= new QGroupBox("视角变换");
	QVBoxLayout *viewLayout = new QVBoxLayout();

	QLabel		*verticalLabel	 = new QLabel("垂直");
	QLabel		*horizontalLabel = new QLabel("水平");
	QHBoxLayout *verticalLayout	 = new QHBoxLayout();
	QSlider		*verticalSlider	 = new QSlider(Qt::Horizontal);
	verticalSlider->setRange(1, 9);
	verticalSlider->setValue(5);
	QSpinBox *verticalSpinBox = new QSpinBox();
	verticalSpinBox->setRange(1, 9);
	verticalSpinBox->setValue(5);
	verticalSpinBox->setSingleStep(1);
	verticalLayout->addWidget(verticalSlider);
	verticalLayout->addWidget(verticalSpinBox);

	QHBoxLayout *horizontalLayout = new QHBoxLayout();
	QSlider		*horizontalSlider = new QSlider(Qt::Horizontal);
	horizontalSlider->setRange(1, 9);
	horizontalSlider->setValue(5);
	QSpinBox *horizontalSpinBox = new QSpinBox();
	horizontalSpinBox->setRange(1, 9);
	horizontalSpinBox->setValue(5);
	horizontalSpinBox->setSingleStep(1);
	horizontalLayout->addWidget(horizontalSlider);
	horizontalLayout->addWidget(horizontalSpinBox);

	viewLayout->addWidget(verticalLabel);
	viewLayout->addLayout(verticalLayout);
	viewLayout->addWidget(horizontalLabel);
	viewLayout->addLayout(horizontalLayout);
	viewGroup->setLayout(viewLayout);
	// TODO: 值传递给LFP
	connect(verticalSlider, &QSlider::valueChanged, verticalSpinBox,
			[=](int value) { verticalSpinBox->setValue(value); });
	connect(verticalSpinBox, QOverload<int>::of(&QSpinBox::valueChanged),
			verticalSlider,
			[=](int value) { verticalSlider->setValue(value); });
	connect(horizontalSlider, &QSlider::valueChanged, horizontalSpinBox,
			[=](int value) { horizontalSpinBox->setValue(value); });
	connect(horizontalSpinBox, QOverload<int>::of(&QSpinBox::valueChanged),
			horizontalSlider,
			[=](int value) { horizontalSlider->setValue(value); });

	return viewGroup;
}

// 3. 重聚焦组
QGroupBox *MainWindow::setupRefocusGroup() {
	QGroupBox	*refocusGroup  = new QGroupBox("重聚焦");
	QVBoxLayout *refocusLayout = new QVBoxLayout();

	QHBoxLayout *viewsLayout = new QHBoxLayout();
	QLabel		*viewsLabel	 = new QLabel("视角数");
	QLabel		*alphaLabel	 = new QLabel("alpha");
	QSlider		*viewsSlider = new QSlider(Qt::Horizontal);
	viewsSlider->setRange(1, 9);
	viewsSlider->setValue(9);
	QSpinBox *viewsSpinBox = new QSpinBox();
	viewsSpinBox->setRange(1, 9);
	viewsSpinBox->setValue(9);
	viewsSpinBox->setSingleStep(1);
	viewsLayout->addWidget(viewsSlider);
	viewsLayout->addWidget(viewsSpinBox);

	QHBoxLayout *alphaLayout = new QHBoxLayout();
	QSlider		*alphaSlider = new QSlider(Qt::Horizontal);
	alphaSlider->setRange(0, 300); // 0.00~3.00 映射为 0~300
	alphaSlider->setValue(100);	   // 默认值 1.50
	QDoubleSpinBox *alphaSpinBox = new QDoubleSpinBox();
	alphaSpinBox->setRange(0.0, 3.0);  // 范围
	alphaSpinBox->setDecimals(2);	   // 保留x位小数
	alphaSpinBox->setValue(1.0);	   // 默认值
	alphaSpinBox->setSingleStep(0.01); // 步长
	alphaLayout->addWidget(alphaSlider);
	alphaLayout->addWidget(alphaSpinBox);

	refocusLayout->addWidget(viewsLabel);
	refocusLayout->addLayout(viewsLayout);
	refocusLayout->addWidget(alphaLabel);
	refocusLayout->addLayout(alphaLayout);
	refocusGroup->setLayout(refocusLayout);

	// TODO: 值传递给LFP
	connect(viewsSlider, &QSlider::valueChanged, viewsSpinBox,
			[=](int value) { viewsSpinBox->setValue(value); });
	connect(viewsSpinBox, QOverload<int>::of(&QSpinBox::valueChanged),
			viewsSlider, [=](int value) { viewsSlider->setValue(value); });
	connect(alphaSlider, &QSlider::valueChanged, alphaSpinBox, [=](int value) {
		alphaSpinBox->setValue(value / 100.0); // 将0~300映射为0.00~3.00
	});
	connect(alphaSpinBox, QOverload<double>::of(&QDoubleSpinBox::valueChanged),
			alphaSlider, [=](double value) {
				alphaSlider->setValue(
					static_cast<int>(value * 100)); // 将0.00~3.00映射为0~300
			});
	return refocusGroup;
}

// 4. 超分辨组
QGroupBox *MainWindow::setupSRGroup() {
	QGroupBox	*srGroup  = new QGroupBox("超分辨");
	QVBoxLayout *srLayout = new QVBoxLayout();
	srGroup->setLayout(srLayout);

	return srGroup;
}

// 5. 深度估计组
QGroupBox *MainWindow::setupDEGroup() {
	QGroupBox	*deGroup  = new QGroupBox("深度估计");
	QVBoxLayout *deLayout = new QVBoxLayout();
	deGroup->setLayout(deLayout);

	return deGroup;
}

}; // namespace Ui
