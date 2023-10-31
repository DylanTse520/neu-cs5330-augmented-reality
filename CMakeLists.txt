cmake_minimum_required(VERSION 3.9)

project(ar)

set(CMAKE_CXX_STANDARD 11)

add_executable(ar ./src/main.cpp ./src/util.cpp ./src/util.hpp ./src/csv_util.cpp ./src/csv_util.h)

find_package(OpenCV REQUIRED)

target_link_libraries(ar ${OpenCV_LIBRARIES})