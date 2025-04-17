#include <QtCore/qdebug.h>
#include <QtCore/qlogging.h>
#include <QtCore/qnamespace.h>
#include <QtCore/qobject.h>
#include <QtCore/qobjectdefs.h>
#include <QtCore/qthread.h>
#include <QtWidgets/qmainwindow.h>

#include <QApplication>
#include <QMetaMethod>
#include <QMetaObject>
#include <QObject>
#include <QPushButton>
#include <QThread>
#include <string>

#include "lfloader.h"
#include "lfprocessor.h"
#include "window_base.h"

int main(int argc, char *argv[]) {
	QApplication app(argc, argv);
	WindowBase	 window;

	// std::string path(argv[1]);
	// bool		isRGB = strcmp(argv[2], "1") == 0 ? true : false;

	// window.worker		= new LFProcessor();
	// QThread *lfp_thread = new QThread();

	// window.worker->moveToThread(lfp_thread);
	// QObject::connect(lfp_thread, &QThread::started, window.worker,
	// 				 &LFProcessor::init_loader, Qt::QueuedConnection);
	// lfp_thread->start();

	// qDebug() << "Main thread:" << window.thread;
	// qDebug() << "LFProcessor thread:" << window.worker->thread();
	// qDebug() << "LFLoader thread:" << window.worker->loader->thread();

	// qDebug() << "Main threadId:" << QThread::currentThreadId();
	// QObject::connect(
	// 	window.button, &QPushButton::clicked, window.worker->loader,
	// 	[worker = window.worker, path, isRGB]() {
	// 		std::cout << "LFProcessor threadId: " << QThread::currentThreadId()
	// 				  << std::endl;
	// 		worker->loader->load(path, isRGB);
	// 	});
	// std::cout << "Main thread: " << QThread::currentThreadId() << std::endl;
	// QObject::connect(
	// 	window.button, &QPushButton::clicked, window.worker,
	// 	[&]() {
	// 		std::cout << "LFProcessor thread: " << QThread::currentThreadId()
	// 				  << std::endl;
	// 		window.worker->loader->invoke(&LFLoader::Core::load, path, isRGB);
	// 	},
	// 	Qt::QueuedConnection);
	// qDebug() << "当前线程：" << QThread::currentThread();
	// qDebug() << "lfp 所在线程：" << lfp->thread();
	// if (lfp->loader->thread()) {
	// 	qDebug() << "lfp->loader 所在线程：" << lfp->loader->thread();
	// }

	// TODO: 解决信号槽lfp与lfloader同线程

	window.show();
	return app.exec();
}