#ifndef LFPROCESSOR_H
#define LFPROCESSOR_H

#include <QObject>

#include "lfloader.h"

class LFProcessor : public QObject {
	Q_OBJECT
   public:
	explicit LFProcessor(QObject *parent = nullptr);
	~LFProcessor();
};

#endif