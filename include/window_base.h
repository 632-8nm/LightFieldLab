#ifndef WINDOW_BASE_H
#define WINDOW_BASE_H
#include <QtCore/qtmetamacros.h>

#include <QMainWindow>
#include <QObject>
#include <QPushButton>
#include <QThread>
#include <QVBoxLayout>

// template <typename Worker>
class WindowBase : public QMainWindow {
	Q_OBJECT
   public:
	WindowBase(QWidget* parent = nullptr) : QMainWindow(parent) {
		QWidget*	 centralWidget = new QWidget(this);
		QVBoxLayout* layout		   = new QVBoxLayout(this);
		button					   = new QPushButton("execute", this);

		layout->addWidget(button);
		centralWidget->setLayout(layout);
		setCentralWidget(centralWidget);

		thread = new QThread(this);
	}
	~WindowBase() {
		if (thread) {
			thread->wait();
			thread->quit();
			if (thread->isRunning()) {
				thread->terminate(); // 强制终止（最后手段）
			}
			delete thread;
			thread = nullptr;
		}
	}
	QThread*	 thread;
	QPushButton* button;
};

#endif