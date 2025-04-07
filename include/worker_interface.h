#ifndef WORKER_INTERFACE
#define WORKER_INTERFACE

#include <QtCore/qtmetamacros.h>

#include <QObject>

class WorkerInterface : public QObject {
	Q_OBJECT
   public:
	// template <typename Func, typename... Args>
	// void invoke(Func&& func, Args&&... args) {
	// 	QMetaObject::invokeMethod(this, std::forward<Func>(func),
	// 							  Qt::QueuedConnection,
	// 							  Q_ARG(Args, std::forward<Args>(args))...);
	// }

	template <typename Func>
	void invokeAsync(Func&& func) {
		// 利用 lambda 将任务投递到当前线程（或指定线程）
		QMetaObject::invokeMethod(
			this, [f = std::forward<Func>(func)]() mutable { f(); },
			Qt::QueuedConnection);
	}
};

#endif