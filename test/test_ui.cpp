#include <QApplication>

#include "ui.h"

int main(int argc, char *argv[]) {
	QApplication   app(argc, argv);
	Ui::MainWindow window = Ui::MainWindow();
	window.setupUi(&window);
	window.show();
	return app.exec();
}
