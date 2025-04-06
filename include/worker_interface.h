#ifndef WORKER_INTERFACE
#define WORKER_INTERFACE

#include <QtCore/qtmetamacros.h>

#include <QObject>

class WorkerInterface : public QObject {
	Q_OBJECT
   public:
	template <typename Func, typename... Args>
	void invoke(Func&& func, Args&&... args) {
		QMetaObject::invokeMethod(this, std::forward<Func>(func),
								  Q_ARG(Args, std::forward<Args>(args))...);
	}
};

#endif