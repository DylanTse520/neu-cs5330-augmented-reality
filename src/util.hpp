/*
  Yixiang Xie
  Fall 2023
  CS 5330
*/

#include <string>
#include <opencv2/opencv.hpp>

int fix_thresholding(const cv::Mat &src, cv::Mat &dst, int threshold = 100);
int adaptive_thresholding(const cv::Mat &src, cv::Mat &dst);
int visualize_segmentation(const cv::Mat &labels, const cv::Mat &stats, std::vector<cv::Vec3b> &colors, std::vector<int> &sortedLabels, cv::Mat &dst, bool debug = false);
int draw_axis_of_least_central_moment(const cv::Point2f &centroid, double angle, cv::Mat &dst);
int calculate_oriented_bounding_box(const cv::Mat &mask, const cv::Point2d &centroid, double angle, std::vector<cv::Point2d> &corners, std::vector<double> &size);
int draw_oriented_bounding_box(std::vector<cv::Point2d> &corners, double percentFilled, double ratio, int number, std::string label, cv::Mat &dst);
std::string get_object_label_nearest_neighbor(std::vector<std::string> &dbLabels, std::vector<std::vector<float>> &dbFeatures, double percentFilled, double ratio);
std::string get_object_label_k_nearest_neighbor(std::vector<std::string> &dbLabels, std::vector<std::vector<float>> &dbFeatures, double percentFilled, double ratio, int k = 3);
std::string get_image_name(std::string foldername, std::string imageType);
std::string get_label_from_zenity();
