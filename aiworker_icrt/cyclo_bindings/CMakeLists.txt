# Build the cyclo_control Python bindings.
#
# Must be built in the SAME environment cyclo_control itself was built in --
# the ROS 2 Jazzy container (docker/DockerFile.{arm64,x86}). The installed
# libcyclo_motion_controller_core.so links against pinocchio 4.1.0, coal 3.0.3,
# urdfdom 4.0, osqp and OsqpEigen 0.10.3, none of which resolve on the bare
# host. See build.sh.
cmake_minimum_required(VERSION 3.16)
project(cyclo_py LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Root of cyclo_control's colcon install tree; override with -DCYCLO_INSTALL=...
set(CYCLO_INSTALL "${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/cyclo_control/install"
    CACHE PATH "cyclo_control colcon install prefix")

find_package(pybind11 REQUIRED)
find_package(Eigen3 REQUIRED)

set(CORE "${CYCLO_INSTALL}/cyclo_motion_controller_core")
set(OSQP_EIGEN "${CYCLO_INSTALL}/osqp_eigen_vendor/opt/osqp_eigen_vendor")

find_library(CYCLO_CORE_LIB cyclo_motion_controller_core
             PATHS "${CORE}/lib" NO_DEFAULT_PATH REQUIRED)
find_library(OSQP_EIGEN_LIB OsqpEigen
             PATHS "${OSQP_EIGEN}/lib" NO_DEFAULT_PATH REQUIRED)

pybind11_add_module(cyclo_py cyclo_py.cpp)

# Consumer headers are written as "common/...", "kinematics/..." relative to the
# cyclo_motion_controller_core directory, so that directory is the include root.
target_include_directories(cyclo_py PRIVATE
    "${CORE}/include/cyclo_motion_controller_core"
    "${CORE}/include"
    "${OSQP_EIGEN}/include")

target_link_libraries(cyclo_py PRIVATE
    ${CYCLO_CORE_LIB} ${OSQP_EIGEN_LIB} Eigen3::Eigen)

# Without this the extension imports but cannot find the cyclo/osqp .so files.
set_target_properties(cyclo_py PROPERTIES
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH "${CORE}/lib;${OSQP_EIGEN}/lib")
