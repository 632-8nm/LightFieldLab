#include <iostream>

#include "refocus.h"

int main(int argc, char* argv[]) {
	std::cout << "Testing Refocus class" << std::endl;
	Refocus* rfc = new Refocus();
	delete rfc;
	return 0;
}