#ifndef WORKER_INTERFACE_H
#define WORKER_INTERFACE_H

#include <QtCore/qtmetamacros.h>

#include <QObject>

class WorkerInterface : public QObject {
	Q_OBJECT
   public:
	template <typename Func, typename... Args>
	void invoke(Func&& func, Args&&... args) {
		QMetaObject::invokeMethod(this, std::forward<Func>(func),
								  Qt::QueuedConnection,
								  Q_ARG(Args, std::forward<Args>(args))...);
	}
};

#endif