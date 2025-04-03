#ifndef REFOCUS_H
#define REFOCUS_H
#include <QObject>
#include <QTHread>
class Refocus : public QObject {
	Q_OBJECT
   public:
	explicit Refocus(QObject* parent = nullptr);
	~Refocus();
};
#endif