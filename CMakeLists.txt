cmake_minimum_required(VERSION 3.9)

project(calibrateAndAR)

set(CMAKE_CXX_STANDARD 11)

add_executable(calibrate ./src/calibrate.cpp ./src/util.cpp ./src/util.hpp ./src/csv_util.cpp ./src/csv_util.h)
add_executable(ar ./src/ar.cpp ./src/util.cpp ./src/util.hpp ./src/csv_util.cpp ./src/csv_util.h)
add_executable(feature ./src/feature.cpp ./src/util.cpp ./src/util.hpp)

find_package(OpenCV REQUIRED)

target_link_libraries(calibrate ${OpenCV_LIBRARIES})
target_link_libraries(ar ${OpenCV_LIBRARIES})
target_link_libraries(feature ${OpenCV_LIBRARIES})