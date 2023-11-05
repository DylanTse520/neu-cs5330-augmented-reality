/*
  Yixiang Xie
  Fall 2023
  CS 5330
*/

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

std::string get_image_name(std::string foldername, std::string imageType);
int print_mat(cv::Mat mat);
int mat_to_vector(cv::Mat cameraMatrix, cv::Mat distCoeffs, std::vector<double> &vec);
int vector_to_mat(std::vector<double> vec, cv::Mat &cameraMatrix, cv::Mat &distCoeffs);
int read_object_data(std::string filename, std::vector<cv::Point3f> &vertices, std::vector<std::vector<int>> &faces);
int draw_corners(cv::Mat cameraMatrix, cv::Mat distCoeffs, cv::Vec3d rvec, cv::Vec3d tvec, cv::Mat &frame);
int draw_object(cv::Mat cameraMatrix, cv::Mat distCoeffs, cv::Vec3d rvec, cv::Vec3d tvec, std::vector<cv::Point3f> vertices, std::vector<std::vector<int>> faces, cv::Mat &frame);
