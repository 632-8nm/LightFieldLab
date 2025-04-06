cmake_minimum_required(VERSION 3.10)
project(test)
# message(STATUS "CMAKE_PREFIX_PATH: ${CMAKE_PREFIX_PATH}")

set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

# 定义资源目录
set(RESOURCE_DIR "${CMAKE_SOURCE_DIR}/data")
set(CMAKE_BUILD_TYPE "Debug")
# set(CMAKE_PREFIX_PATH /usr/local/opencv/${CMAKE_BUILD_TYPE})
# message(STATUS "CMAKE_PREFIX_PATH: ${CMAKE_PREFIX_PATH}")
message("==================================================================")
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs highgui)
find_package(QT NAMES Qt6 Qt5 REQUIRED COMPONENTS Widgets)
find_package(Qt${QT_VERSION_MAJOR} REQUIRED COMPONENTS Widgets)

# test_metal
add_executable(test_metal test_metal.cpp)
target_link_libraries(test_metal ${OpenCV_LIBS})

# test_loader
add_executable(test_loader
    test_loader.cpp
    ${CMAKE_SOURCE_DIR}/src/lfloader.cpp
    ${CMAKE_SOURCE_DIR}/include/lfloader.h
)
target_include_directories(test_loader PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(test_loader PRIVATE Qt${QT_VERSION_MAJOR}::Widgets ${OpenCV_LIBS})

# test_refocuser
add_executable(test_refocuser
    test_refocuser.cpp
    ${CMAKE_SOURCE_DIR}/src/lfrefocuser.cpp
    ${CMAKE_SOURCE_DIR}/src/lfloader.cpp
    ${CMAKE_SOURCE_DIR}/include/lfrefocuser.h
    ${CMAKE_SOURCE_DIR}/include/lfloader.h
    ${CMAKE_SOURCE_DIR}/include/worker_interface.h
)
target_include_directories(test_refocuser PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(test_refocuser PRIVATE Qt${QT_VERSION_MAJOR}::Widgets ${OpenCV_LIBS})
# message(STATUS "OpenCV_LIBS: ${OpenCV_LIBS}")
