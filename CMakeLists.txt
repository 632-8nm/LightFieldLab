cmake_minimum_required(VERSION 3.16)

project(LightFieldLab VERSION 0.1 LANGUAGES CXX)

set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
# set(CMAKE_AUTOUIC ON CACHE STRING "CMAKE_AUTOUIC")
# set(CMAKE_AUTOMOC ON CACHE STRING "CMAKE_AUTOMOC")
# set(CMAKE_AUTORCC ON CACHE STRING "CMAKE_AUTORCC")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE "Debug")
set(CMAKE_INSTALL_PREFIX "${PROJECT_SOURCE_DIR}/install")
# set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_BUILD_TYPE}/bin)
# set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_BUILD_TYPE}/lib)
# set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${CMAKE_BUILD_TYPE}/lib)


if(WIN32) # Windows
    set(CMAKE_PREFIX_PATH
        "D:/Qt/6.7.2/msvc2019_64;D:/opencv/install/x64/vc17/lib" CACHE STRING "Path to Qt and OpenCV")
else() # Linux/MacOS
    set(CMAKE_PREFIX_PATH
        "~/Qt/6.8.3/macos/lib/cmake;/usr/local/opencv/${CMAKE_BUILD_TYPE}" CACHE STRING "Path to Qt and OpenCV")
endif()

add_subdirectory(test)

find_package(QT NAMES Qt6 Qt5 REQUIRED COMPONENTS Widgets)
find_package(Qt${QT_VERSION_MAJOR} REQUIRED COMPONENTS Widgets)
find_package(OpenCV REQUIRED core highgui imgproc imgcodecs)

# file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS "src/*.cpp")
# file(GLOB_RECURSE HEADERS CONFIGURE_DEPENDS "include/*.h")
# add_executable(${PROJECT_NAME} ${SOURCES} ${HEADERS})

set(PROJECT_SOURCES
    src/main.cpp
    src/mainwindow.cpp
    src/ui.cpp
    src/lfprocessor.cpp
    src/lfload.cpp
    src/lfrefocus.cpp
    src/lfsuperres.cpp
    include/mainwindow.h
    include/ui.h
    include/lfdata.h
    include/lfprocessor.h
    include/lfload.h
    include/lfrefocus.h
    include/lfsuperres.h
)
set(BIN_NAME "lightfieldlab")
add_executable(${BIN_NAME} ${PROJECT_SOURCES})
target_include_directories(${BIN_NAME} PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(${BIN_NAME} PRIVATE
    Qt${QT_VERSION_MAJOR}::Widgets
    ${OpenCV_LIBS}
    opencv_dnn_superres
)

# set(BUNDLE_PATH "${CMAKE_INSTALL_PREFIX}/lightfieldlab.app")
# set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
# set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
# set(CMAKE_INSTALL_RPATH "@executable_path/../lib")

# set_target_properties(${BIN_NAME} PROPERTIES
#     ${BUNDLE_ID_OPTION}
#     MACOSX_BUNDLE_BUNDLE_VERSION ${PROJECT_VERSION}
#     MACOSX_BUNDLE_SHORT_VERSION_STRING ${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}
#     MACOSX_BUNDLE TRUE
#     WIN32_EXECUTABLE TRUE
# )

# include(GNUInstallDirs)
# install(TARGETS ${BIN_NAME}
#     BUNDLE DESTINATION ${CMAKE_INSTALL_PREFIX}
#     LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
#     RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
# )
file(COPY "${CMAKE_SOURCE_DIR}/data" DESTINATION "${CMAKE_BINARY_DIR}")


# if(${CMAKE_SYSTEM_NAME} STREQUAL "Darwin")
#     install(SCRIPT "${PROJECT_SOURCE_DIR}/install.cmake")
# endif()

# info output 
message(STATUS "===============================================================")
message(STATUS "Current system is ${CMAKE_SYSTEM_NAME}")
message(STATUS "CMAKE_BINARY_DIR: ${CMAKE_BINARY_DIR}")
message(STATUS "CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")
message(STATUS "CMAKE_PREFIX_PATH: ${CMAKE_PREFIX_PATH}")
message(STATUS "CMAKE_INSTALL_PREFIX: ${CMAKE_INSTALL_PREFIX}")
message(STATUS "CMAKE_INSTALL_LIBDIR: ${CMAKE_INSTALL_LIBDIR}")
message(STATUS "CMAKE_INSTALL_BINDIR: ${CMAKE_INSTALL_BINDIR}")
message(STATUS "CMAKE_INSTALL_INCLUDEDIR: ${CMAKE_INSTALL_INCLUDEDIR}")
message(STATUS "CMAKE_RUNTIME_OUTPUT_DIRECTORY: ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
message(STATUS "CMAKE_LIBRARY_OUTPUT_DIRECTORY: ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}")
message(STATUS "CMAKE_ARCHIVE_OUTPUT_DIRECTORY: ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY}")
message(STATUS "OpenCV_DIR: ${OpenCV_DIR}")
message(STATUS "OpenCV_VERSION: ${OpenCV_VERSION}")
message(STATUS "OpenCV_LIB_DIR: ${OpenCV_LIB_DIR}")
message(STATUS "OpenCV_LIBS: ${OpenCV_LIBS}")
message(STATUS "OpenCV_INCLUDE_DIRS: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "QT_VERSION: ${QT_VERSION}")
message(STATUS "===============================================================")
