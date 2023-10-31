/*
  Yixiang Xie
  Fall 2023
  CS 5330
*/

#include "util.hpp"

// perform thresholding with a given fixed threshold
// pixels with values greater than the threshold are set to 255
// src: the source image, which is a grayscale image
// dst: the destination image
// threshold: the threshold value
// return: 0 if successful, non-zero if there is an error
int fix_thresholding(const cv::Mat &src, cv::Mat &dst, int threshold)
{
  // error checking
  if (src.data == NULL)
  {
    printf("error: fix_thresholding() src image is empty\n");
    return (-1);
  }

  // error checking
  if (src.channels() != 1)
  {
    printf("error: fix_thresholding() src image is not grayscale\n");
    return (-2);
  }

  // error checking
  if (src.depth() != CV_8U)
  {
    printf("error: fix_thresholding() src image is not 8-bit\n");
    return (-3);
  }

  // error checking
  if (threshold < 0 || threshold > 255)
  {
    printf("error: fix_thresholding() threshold is out of range\n");
    return (-4);
  }

  // create a destination image
  dst = cv::Mat(src.rows, src.cols, CV_8UC1);

  // perform thresholding
  for (int y = 0; y < src.rows; y++)
  {
    // the src and dst pointers for this row
    const uchar *src_ptr = src.ptr<uchar>(y);
    uchar *dst_ptr = dst.ptr<uchar>(y);

    for (int x = 0; x < src.cols; x++)
    {
      // if the pixel value is greater than the threshold, set it to 255
      // otherwise, set it to 0
      if (src_ptr[x] > threshold)
      {
        dst_ptr[x] = 255;
      }
      else
      {
        dst_ptr[x] = 0;
      }
    }
  }

  return (0);
}

// perform adaptive thresholding by running a k-means (k = 2) clustering algorithm
// on a random sample of 1/4 of the pixels from the image to decide the threshold
// src: the source image, which is a grayscale image
// dst: the destination image
// return: 0 if successful, non-zero if there is an error
int adaptive_thresholding(const cv::Mat &src, cv::Mat &dst)
{
  // error checking
  if (src.data == NULL)
  {
    printf("error: adaptive_thresholding() src image is empty\n");
    return (-1);
  }

  // error checking
  if (src.channels() != 1)
  {
    printf("error: adaptive_thresholding() src image is not grayscale\n");
    return (-2);
  }

  // error checking
  if (src.depth() != CV_8U)
  {
    printf("error: adaptive_thresholding() src image is not 8-bit\n");
    return (-3);
  }

  // clone the src and reshape to a vector
  cv::Mat reshaped = src.clone();
  reshaped.reshape(1, 1);

  // shuffle the vector
  cv::randShuffle(reshaped);

  // get 1/4 of the vector
  cv::Mat shuffled = reshaped(cv::Rect(0, 0, reshaped.cols / 4, reshaped.rows));

  // convert the vector to CV_32F
  cv::Mat dataMat(shuffled);
  dataMat.convertTo(dataMat, CV_32F);

  // perform k-means clustering on the data value
  cv::Mat labels;
  cv::Mat centers;
  int attempts = 3;

  cv::TermCriteria criteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 10, 1.0);
  cv::kmeans(dataMat, 2, labels, criteria, attempts, cv::KMEANS_PP_CENTERS, centers);

  // get the mean value of the centers
  float mean = (centers.at<float>(0) + centers.at<float>(1)) / 2.0f;

  // use the mean value to threshold the image
  fix_thresholding(src, dst, mean);

  return (0);
}

// visualize the segmented image,
// which maps the 7 biggest connected components to 7 different colors
// labels: the labels of the connected components
// stats: the statistics of the connected components
// dst: the destination image
// return: 0 if successful, non-zero if there is an error
int visualize_segmentation(const cv::Mat &labels, const cv::Mat &stats, std::vector<cv::Vec3b> &colors, std::vector<int> &sortedLabels, cv::Mat &dst, bool debug)
{
  // error checking
  if (labels.data == NULL)
  {
    printf("error: visualize_segmentation() labels image is empty\n");
    return (-1);
  }

  // error checking
  if (stats.data == NULL)
  {
    printf("error: visualize_segmentation() stats image is empty\n");
    return (-2);
  }

  // init sorted labels
  sortedLabels = std::vector<int>(stats.rows);
  for (int i = 0; i < sortedLabels.size(); ++i)
  {
    sortedLabels[i] = i;
  }
  // sort indices by area size of stats
  int columnToSortBy = cv::CC_STAT_AREA;
  std::sort(sortedLabels.begin(), sortedLabels.end(), [&stats, columnToSortBy](int i, int j)
            { return stats.at<int>(i, columnToSortBy) > stats.at<int>(j, columnToSortBy); });

  // map first 7 nonzero labels to colors
  std::map<int, cv::Vec3b> colorMap;
  for (int i = 0; i < sortedLabels.size(); ++i)
  {
    // get the label
    int label = sortedLabels[i];
    // if the label is 0 (background), skip it
    if (label == 0)
    {
      continue;
    }
    // if the area is smaller than 1000, stop
    if (stats.at<int>(label, cv::CC_STAT_AREA) < 1000)
    {
      break;
    }
    // if colors not empty, pop the first color in colors
    cv::Vec3b color;
    if (!colors.empty())
    {
      color = colors.back();
      colors.pop_back();
    }
    else
    {
      break;
    }
    // map the label to the color
    colorMap[label] = color;
  }

  // if debug is true, print the color map
  if (debug)
  {
    printf("color map:\n");
    for (auto const &pair : colorMap)
    {
      printf("  %d: (%d, %d, %d)\n", pair.first, pair.second[0], pair.second[1], pair.second[2]);
    }
  }

  // create a destination image, initialized to black
  dst = cv::Mat(labels.rows, labels.cols, CV_8UC3, cv::Scalar(0, 0, 0));

  // color output image based on labels
  for (int i = 0; i < dst.rows; ++i)
  {
    // get the row pointers
    int const *labelsRowPtr = labels.ptr<int>(i);
    cv::Vec3b *segmentedImageRowPtr = dst.ptr<cv::Vec3b>(i);
    for (int j = 0; j < dst.cols; ++j)
    {
      // get the label
      int label = labelsRowPtr[j];
      // if the label is the biggest component
      if (label == 0)
      {
        continue;
      }
      // if the label is not mapped to a color
      else if (colorMap.find(label) == colorMap.end())
      {
        continue;
      }
      else
      {
        // get the color
        cv::Vec3b color = colorMap[label];
        // set the pixel
        segmentedImageRowPtr[j] = color;
      }
    }
  }

  return (0);
}

// draw the axis of the least central moment
// centroid: the centroid of the component
// angle: the angle of the axis
// dst: the destination image
// return: 0 if successful, non-zero if there is an error
int draw_axis_of_least_central_moment(const cv::Point2f &centroid, double angle, cv::Mat &dst)
{
  // error checking
  if (dst.data == NULL)
  {
    printf("error: draw_axis_of_least_central_moment() dst image is empty\n");
    return (-1);
  }

  // draw the centroid
  cv::circle(dst, centroid, 5, cv::Scalar(255, 255, 255), -1);

  // draw the central axis
  cv::Point2d p1(centroid.x + 100 * cos(angle), centroid.y + 100 * sin(angle));
  cv::Point2d p2(centroid.x, centroid.y);
  cv::line(dst, p1, p2, cv::Scalar(255, 255, 255), 2);

  // draw an arrow at the end of the central axis
  cv::Point2d p3(centroid.x + 100 * cos(angle) - 10 * cos(0.5 - angle), centroid.y + 100 * sin(angle) + 10 * sin(0.5 - angle));
  cv::Point2d p4(centroid.x + 100 * cos(angle) - 10 * cos(0.5 + angle), centroid.y + 100 * sin(angle) - 10 * sin(0.5 + angle));
  cv::line(dst, p1, p3, cv::Scalar(255, 255, 255), 2);
  cv::line(dst, p1, p4, cv::Scalar(255, 255, 255), 2);

  return (0);
}

// calculate the oriented bounding box of component
// mask: the mask of the component
// centroid: the centroid of the component
// angle: the angle of the axis
// corners: the corners of the bounding box
// size: the height and width of the bounding box
// return: 0 if successful, non-zero if there is an error
int calculate_oriented_bounding_box(const cv::Mat &mask, const cv::Point2d &centroid, double angle, std::vector<cv::Point2d> &corners, std::vector<double> &size)
{
  // error checking
  if (mask.data == NULL)
  {
    printf("error: calculate_oriented_bounding_box() mask image is empty\n");
    return (-1);
  }

  // the bounding box coordinates
  double x_max = -DBL_MAX, x_min = DBL_MAX, y_max = -DBL_MAX, y_min = DBL_MAX;

  // iterate through the mask
  for (int y = 0; y < mask.rows; y++)
  {
    // the mask and dst pointers for this row
    const uchar *mask_ptr = mask.ptr<uchar>(y);

    for (int x = 0; x < mask.cols; x++)
    {
      // if the pixel is not part of the component, skip it
      if (mask_ptr[x] == 0)
      {
        continue;
      }

      // rotate the coordinates around the centroid
      double x_rotated = (y - centroid.y) * sin(angle) + (x - centroid.x) * cos(angle);
      double y_rotated = (y - centroid.y) * cos(angle) - (x - centroid.x) * sin(angle);

      // update the bounding box coordinates
      if (x_rotated > x_max)
      {
        x_max = x_rotated;
      }
      if (x_rotated < x_min)
      {
        x_min = x_rotated;
      }
      if (y_rotated > y_max)
      {
        y_max = y_rotated;
      }
      if (y_rotated < y_min)
      {
        y_min = y_rotated;
      }
    }
  }

  // calculate the corners of the bounding box
  cv::Point2d p1(centroid.x + x_max * cos(angle) - y_max * sin(angle), centroid.y + x_max * sin(angle) + y_max * cos(angle));
  cv::Point2d p2(centroid.x + x_max * cos(angle) - y_min * sin(angle), centroid.y + x_max * sin(angle) + y_min * cos(angle));
  cv::Point2d p3(centroid.x + x_min * cos(angle) - y_min * sin(angle), centroid.y + x_min * sin(angle) + y_min * cos(angle));
  cv::Point2d p4(centroid.x + x_min * cos(angle) - y_max * sin(angle), centroid.y + x_min * sin(angle) + y_max * cos(angle));

  // add the corners to the corners vector
  corners.push_back(p1);
  corners.push_back(p2);
  corners.push_back(p3);
  corners.push_back(p4);

  // calculate the height and width of the bounding box
  double height = x_max - x_min;
  double width = y_max - y_min;

  // add the height and width to the size vector
  size.push_back(height);
  size.push_back(width);

  return (0);
}

// draw the oriented bounding box
// corners: the corners of the bounding box
// percentFilled: the percent filled of the component
// ratio: the height/width ratio of the component
// dst: the destination image
// return: 0 if successful, non-zero if there is an error
int draw_oriented_bounding_box(std::vector<cv::Point2d> &corners, double percentFilled, double ratio, int number, std::string label, cv::Mat &dst)
{
  // error checking
  if (dst.data == NULL)
  {
    printf("error: draw_oriented_bounding_box() dst image is empty\n");
    return (-1);
  }

  // draw the bounding box
  cv::line(dst, corners[0], corners[1], cv::Scalar(255, 255, 255), 1);
  cv::line(dst, corners[1], corners[2], cv::Scalar(255, 255, 255), 1);
  cv::line(dst, corners[2], corners[3], cv::Scalar(255, 255, 255), 1);
  cv::line(dst, corners[3], corners[0], cv::Scalar(255, 255, 255), 1);

  // label the object number
  std::string text = "Object " + std::to_string(number);
  cv::putText(dst, text, corners[1] + cv::Point2d(10, -5), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 255));

  // label the object label
  text = "Label: " + label;
  cv::putText(dst, text, corners[1] + cv::Point2d(10, 20), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 255));

  // label the percent filled
  text = "Percent filled: " + (std::to_string(percentFilled * 100)).erase(5) + "%";
  cv::putText(dst, text, corners[1] + cv::Point2d(10, 45), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 255));

  // label the height/width ratio
  text = "Height/width ratio: " + (std::to_string(ratio)).erase(4);
  cv::putText(dst, text, corners[1] + cv::Point2d(10, 70), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 255));

  return (0);
}

// calculate the euclidean distance and return the nearest neighbor label
// dbLabels: the labels of the objects in the database
// dbFeatures: the features of the objects in the database
// percentFilled: the percent filled of the component
// ratio: the height/width ratio of the component
// return: the label of the closest object
std::string get_object_label_nearest_neighbor(std::vector<std::string> &dbLabels, std::vector<std::vector<float>> &dbFeatures, double percentFilled, double ratio)
{
  // if the database is empty, return "Unknown"
  if (dbLabels.size() == 0)
  {
    return ("Unknown");
  }

  // iterate through the database and calculate the euclidean distance
  std::vector<double> feature = {percentFilled, ratio};
  double minDistance = DBL_MAX;
  std::string objectLabel = "";

  // compute the mean of the dataset
  float mean = 0;
  for (int i = 0; i < dbFeatures.size(); i++)
  {
    for (int j = 0; j < dbFeatures[i].size(); j++)
    {
      mean += dbFeatures[i][j];
    }
  }
  mean /= (dbFeatures.size() * dbFeatures[0].size());
  
  // compute the standard deviation of the dataset
  float std = 0;
  for (int i = 0; i < dbFeatures.size(); i++)
  {
    for (int j = 0; j < dbFeatures[i].size(); j++)
    {
      std += (dbFeatures[i][j] - mean) * (dbFeatures[i][j] - mean);
    }
  }
  std /= (dbFeatures.size() * dbFeatures[0].size());
  std = sqrt(std);

  for (int j = 0; j < dbFeatures.size(); j++)
  {
    float distance = 0;
    for (int k = 0; k < dbFeatures[j].size(); k++)
    {
      distance += (dbFeatures[j][k] - feature[k]) * (dbFeatures[j][k] - feature[k]);
    }
    distance /= std;
    if (distance < minDistance)
    {
      minDistance = distance;
      objectLabel = dbLabels[j];
    }
  }

  // if the distance is greater than 0.5, return "Unknown"
  if (minDistance > 0.5)
  {
    return ("Unknown");
  }

  // return the label of the closest object
  return (objectLabel);
}

// calculate the euclidean distance and return the label of the nearest group of neighbors.
// distance is calculated by the average distance of the k nearest neighbors
// dbLabels: the labels of the objects in the database
// dbFeatures: the features of the objects in the database
// percentFilled: the percent filled of the component
// ratio: the height/width ratio of the component
// k: the number of nearest neighbors
// return: the label of the closest object
std::string get_object_label_k_nearest_neighbor(std::vector<std::string> &dbLabels, std::vector<std::vector<float>> &dbFeatures, double percentFilled, double ratio, int k)
{
  // if the database is empty, return "Unknown"
  if (dbLabels.size() == 0)
  {
    return ("Unknown");
  }

  // iterate through the database and construct a map of labels and features
  std::map<std::string, std::vector<std::vector<float>>> labelMap;
  for (int i = 0; i < dbLabels.size(); i++)
  {
    labelMap[dbLabels[i]].push_back(dbFeatures[i]);
  }

  // construct the feature vector
  std::vector<double> feature = {percentFilled, ratio};

  double minDistance = DBL_MAX;
  std::string objectLabel = "";

  // iterate through the map
  for (auto const &pair : labelMap)
  {
    // for each label, calculate all the distances of the features
    std::vector<double> distances;
    for (int i = 0; i < pair.second.size(); i++)
    {
      float distance = 0;
      for (int j = 0; j < pair.second[i].size(); j++)
      {
        distance += (pair.second[i][j] - feature[j]) * (pair.second[i][j] - feature[j]);
      }
      distances.push_back(distance);
    }

    // sort the distances in ascending order
    std::sort(distances.begin(), distances.end());

    // calculate the average distance of the k nearest neighbors
    double distance = 0;
    for (int i = 0; i < k; i++)
    {
      distance += distances[i];
    }
    distance /= k;

    if (distance < minDistance)
    {
      minDistance = distance;
      objectLabel = pair.first;
    }
  }

  // if the distance is greater than 0.5, return "Unknown"
  if (minDistance > 0.5)
  {
    return ("Unknown");
  }

  // return the label of the closest object
  return (objectLabel);
}

// get the image name
// foldername: the folder name
// imageType: the image type
// return: the image name
std::string get_image_name(std::string foldername, std::string imageType)
{
  // get the current time
  time_t rawtime;
  time(&rawtime);

  // convert the time to local time
  struct tm *timeinfo;
  timeinfo = localtime(&rawtime);

  // store current time
  char currentTime[256];
  strftime(currentTime, sizeof(currentTime), "%Y-%m-%d_%H-%M-%S", timeinfo);

  // create the filename
  std::string filename = foldername + imageType + "_" + currentTime + ".jpg";

  return (filename);
}

// get user input from zenity
// title: the title of the zenity window
// text: the text of the zenity window
// return: the user input
std::string get_label_from_zenity()
{
  // run the zenity command and open a pipe to capture its output
  std::string command = "zenity --forms --title=\"Input label\" --text=\"Input the object number and label\" --add-entry=\"Object number\" --add-entry=\"Object label\"";
  FILE *pipe = popen(command.c_str(), "r");

  // error checking
  if (!pipe)
  {
    printf("error: popen() failed!");
    return "";
  }

  // create a buffer to read the output
  char buffer[128];
  std::string result;

  // read the output of zenity and store it in the result string
  while (fgets(buffer, sizeof(buffer), pipe) != nullptr)
  {
    result += buffer;
  }

  // close the pipe
  pclose(pipe);

  // strip newline characters from the result
  result.erase(result.find_last_not_of("\n") + 1);

  // return the result
  return result;
}