cmake_minimum_required(VERSION 3.16)

project(LightFieldLab VERSION 0.1 LANGUAGES CXX)

set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE "Debug")
set(CMAKE_INSTALL_PREFIX "${PROJECT_SOURCE_DIR}/install")

if(WIN32) # Windows
    set(CMAKE_PREFIX_PATH
        "D:/Qt/6.7.2/msvc2019_64"
        "D:/opencv/install/x64/vc17/lib"
    )
else() # Linux/MacOS
    set(CMAKE_PREFIX_PATH "~/Qt/6.8.3/macos/lib/cmake")
    if(${CMAKE_BUILD_TYPE} STREQUAL "Debug")
        list(APPEND CMAKE_PREFIX_PATH "/usr/local/opencv_debug")
    else()
        list(APPEND CMAKE_PREFIX_PATH "/usr/local/opencv_release")
    endif()
endif()

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
    include/mainwindow.h
    include/ui.h
)
set(BIN_NAME "lightfieldlab")
qt_add_executable(${BIN_NAME}
    MANUAL_FINALIZATION
    ${PROJECT_SOURCES}
)

target_include_directories(${BIN_NAME} PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(${BIN_NAME} PRIVATE Qt${QT_VERSION_MAJOR}::Widgets)
target_link_libraries(${BIN_NAME} PRIVATE ${OpenCV_LIBS})

set(BUNDLE_PATH "${CMAKE_INSTALL_PREFIX}/lightfieldlab.app")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)
set(CMAKE_INSTALL_RPATH "@executable_path/../lib")

set_target_properties(${BIN_NAME} PROPERTIES
    ${BUNDLE_ID_OPTION}
    MACOSX_BUNDLE_BUNDLE_VERSION ${PROJECT_VERSION}
    MACOSX_BUNDLE_SHORT_VERSION_STRING ${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}
    MACOSX_BUNDLE TRUE
    WIN32_EXECUTABLE TRUE

    MACOSX_BUNDLE TRUE
    BUNDLE TRUE
    BUNDLE_CONTENTS "${BUNDLE_PATH}/Contents"
)

include(GNUInstallDirs)
install(TARGETS ${BIN_NAME}
    BUNDLE DESTINATION ${CMAKE_INSTALL_PREFIX}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}

)

if(QT_VERSION_MAJOR EQUAL 6)
    qt_finalize_executable(${BIN_NAME})
endif()

install(SCRIPT "${PROJECT_SOURCE_DIR}/install.cmake")
# include(install.cmake)
# enable_bundle_utilities(${BIN_NAME})
# include(BundleUtilities)


# info output 
message(STATUS "==========================")
message(STATUS "Current system is ${CMAKE_SYSTEM_NAME}")
message(STATUS "CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")
message(STATUS "CMAKE_PREFIX_PATH: ${CMAKE_PREFIX_PATH}")
message(STATUS "CMAKE_INSTALL_PREFIX: ${CMAKE_INSTALL_PREFIX}")
message(STATUS "CMAKE_INSTALL_LIBDIR: ${CMAKE_INSTALL_LIBDIR}")
message(STATUS "CMAKE_INSTALL_BINDIR: ${CMAKE_INSTALL_BINDIR}")
message(STATUS "CMAKE_INSTALL_INCLUDEDIR: ${CMAKE_INSTALL_INCLUDEDIR}")
message(STATUS "OpenCV_DIR: ${OpenCV_DIR}")
message(STATUS "OpenCV_VERSION: ${OpenCV_VERSION}")
message(STATUS "OpenCV_LIB_DIR: ${OpenCV_LIB_DIR}")
message(STATUS "OpenCV_LIBS: ${OpenCV_LIBS}")
message(STATUS "OpenCV_INCLUDE_DIRS: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "QT_VERSION: ${QT_VERSION}")
message(STATUS "==========================")
