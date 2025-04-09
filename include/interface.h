#ifndef INTERFACE_H
#define INTERFACE_H

#include <QtCore/qobject.h>
#include <QtCore/qtmetamacros.h>

#include <QMainWindow>
#include <QObject>
#include <QPushButton>
#include <QThread>
#include <QVBoxLayout>
// class Interface : public QObject {
// 	Q_OBJECT
//    public:
// template <typename Func, typename... Args>
// void invoke(Func&& func, Args&&... args) {
// 	QMetaObject::invokeMethod(this, std::forward<Func>(func),
// 							  Qt::QueuedConnection,
// 							  std::forward<Args>(args)...);
// }
// };
template <typename Core>
class WorkerBase : public QObject {
   public:
	explicit WorkerBase(QObject* parent = nullptr)
		: QObject(parent), core(std::make_unique<Core>()) {}
	~WorkerBase() {}

	template <typename Func, typename... Args>
	void invoke(Func&& func, Args&&... args) {
		QMetaObject::invokeMethod(
			this, // 调用 Worker 的匿名槽函数
			[this, func = std::forward<Func>(func),
			 args = std::tuple(std::forward<Args>(args)...)]() mutable {
				// 使用 core.get() 绑定 Core 的 this 指针
				std::apply(
					[&](auto&&... params) {
						(core.get()->*func)(
							std::forward<decltype(params)>(params)...);
					},
					std::move(args));
			},
			Qt::QueuedConnection);
	}

   private:
	std::unique_ptr<Core> core;
};

template <typename Worker>
class WindowBase : public QMainWindow {
   public:
	WindowBase(Worker* worker, QWidget* parent = nullptr)
		: QMainWindow(parent) {
		QWidget*	 centralWidget = new QWidget(this);
		QVBoxLayout* layout		   = new QVBoxLayout(this);
		_button					   = new QPushButton("execute", this);

		layout->addWidget(_button);
		centralWidget->setLayout(layout);
		setCentralWidget(centralWidget);

		_worker = new Worker();
		_thread = new QThread(this);
		_worker->moveToThread(_thread);
		_thread->start();
	}
	~WindowBase() {
		if (_thread) {
			_thread->wait();
			_thread->quit();
			delete _thread;
			_thread = nullptr;
		}
	}

	QThread*	 _thread;
	Worker*		 _worker;
	QPushButton* _button;
};
#endif