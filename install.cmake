# install.cmake
include(BundleUtilities)

# 假设安装路径为 ${CMAKE_INSTALL_PREFIX}/lightfieldlab.app
set(APP_PATH "${CMAKE_INSTALL_PREFIX}/lightfieldlab.app")
set(LIB_PATH "${CMAKE_INSTALL_PREFIX}/lib")

# 修复 bundle 或复制依赖库
fixup_bundle("${APP_PATH}" "" "${LIB_PATH}")
