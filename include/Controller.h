#ifndef CONTROLLER_H
#define CONTROLLER_H

#include <QObject>
#include <memory>

#include "refocus.h"

class Controller : public QObject {
	Q_OBJECT
   public:
	explicit Controller(QObject* parent = nullptr);
	~Controller();

	std::unique_ptr<Refocus> refocuser;

   private:
};

#endif