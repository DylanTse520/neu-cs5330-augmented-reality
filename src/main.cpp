/*
  Yixiang Xie
  Fall 2023
  CS 5330
*/

#include "util.hpp"
#include "csv_util.h"

int main(int argc, char *argv[])
{
  // parse the optional arguments

  // if optional argument has --km, use k-means clustering to threshold the image
  // if optional argument has --knn=[value], set k to [value]
  bool useKMeans = false;
  int k = 1;
  for (int i = 1; i < argc; i++)
  {
    std::string arg = argv[i];
    if (arg.find("--km") == 0)
    {
      useKMeans = true;
    }
    if (arg.find("--knn=") == 0)
    {
      k = std::stoi(arg.substr(6));
    }
  }

  // announce the use of k-means clustering and the value of k
  if (useKMeans)
  {
    printf("using k-means clustering for adaptive thresholding\n");
  }
  if (k > 1)
  {
    printf("using k nearest neighbor for classifier, k = %d\n", k);
  }

  // read the data from the csv file
  std::vector<std::string> dbLabels;
  std::vector<std::vector<float>> dbFeatures;
  read_object_data_csv("../resources/data.csv", dbLabels, dbFeatures);

  // get video stream

  // open the video device
  cv::VideoCapture *vidCap;
  vidCap = new cv::VideoCapture(0);

  // error checking
  if (!vidCap->isOpened())
  {
    printf("error: unable to open video device\n");
    return (-2);
  }

  // get the width and height of frames in the video stream
  cv::Size refS((int)vidCap->get(cv::CAP_PROP_FRAME_WIDTH),
                (int)vidCap->get(cv::CAP_PROP_FRAME_HEIGHT));

  // for all frames
  cv::Mat original;

  while (true)
  {
    // read a frame from the video stream
    *vidCap >> original;

    // error checking
    if (original.empty())
    {
      printf("error: frame is empty\n");
      break;
    }

    // display the frame
    cv::imshow("Original", original);

    // pre-processing

    // blur the image
    cv::Mat blurredImage;
    cv::GaussianBlur(original, blurredImage, cv::Size(3, 3), 0, 0, cv::BORDER_DEFAULT);

    // make strongly colored pixels darker
    cv::Mat hsvImage;
    cv::cvtColor(blurredImage, hsvImage, cv::COLOR_BGR2HSV);
    for (int i = 0; i < hsvImage.rows; i++)
    {
      for (int j = 0; j < hsvImage.cols; j++)
      {
        // get the pixel
        cv::Vec3b pixel = hsvImage.at<cv::Vec3b>(i, j);
        // get the saturation value
        int saturation = pixel[1];
        // if the saturation value is greater than 100, make the pixel darker
        if (saturation > 100)
        {
          pixel[2] = pixel[2] * 0.5;
        }
        // set the pixel
        hsvImage.at<cv::Vec3b>(i, j) = pixel;
      }
    }
    // convert the image back to BGR
    cv::Mat darkerImage;
    cv::cvtColor(hsvImage, darkerImage, cv::COLOR_HSV2BGR);

    // convert the image to grayscale
    cv::Mat grayscaleImage;
    cv::cvtColor(darkerImage, grayscaleImage, cv::COLOR_BGR2GRAY);

    // invert the image, so the foreground pixels are white
    cv::Mat invertedImage;
    cv::bitwise_not(grayscaleImage, invertedImage);

    // thresholding

    cv::Mat binaryImage;

    if (useKMeans)
    {
      // use k-means clustering to threshold the image
      adaptive_thresholding(invertedImage, binaryImage);
    }
    else
    {
      // use fixed thresholding to threshold the image
      fix_thresholding(invertedImage, binaryImage, 155);
    }

    // display the binary image frame
    cv::imshow("Binary", binaryImage);

    // cleaning up

    // clean up the image using morphological closing
    cv::Mat cleanedImage;
    cv::morphologyEx(binaryImage, cleanedImage, cv::MORPH_CLOSE, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5)));

    // display the cleaned image frame
    cv::imshow("Cleaned", cleanedImage);

    // segmentation

    // find the connected components
    cv::Mat labels, stats, centroids;
    int numComponents = cv::connectedComponentsWithStats(cleanedImage, labels, stats, centroids, 8, CV_32S);

    // define seven colors
    std::vector<cv::Vec3b> colors;
    colors.push_back(cv::Vec3b(128, 20, 148)); // purple
    colors.push_back(cv::Vec3b(130, 0, 75));   // indigo
    colors.push_back(cv::Vec3b(240, 100, 20)); // blue
    colors.push_back(cv::Vec3b(20, 220, 20));  // green
    colors.push_back(cv::Vec3b(20, 220, 240)); // yellow
    colors.push_back(cv::Vec3b(20, 100, 240)); // orange
    colors.push_back(cv::Vec3b(20, 20, 240));  // red

    // visualize the segmentation
    cv::Mat segmentedImage;
    std::vector<int> sortedLabels;
    visualize_segmentation(labels, stats, colors, sortedLabels, segmentedImage);

    // display the segmentation visualization frame
    cv::imshow("Segmented", segmentedImage);

    // compute the features and classify the objects

    // the image with features for display
    cv::Mat imageWithFeatures = segmentedImage.clone();

    // the feature map for each object
    std::map<int, std::vector<float>> featureMap;

    // the object number
    int objectNumber = 1;
    // for each region, in descending order of area size
    for (int i = 0; i < sortedLabels.size(); i++)
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
      // if there are more than 7 objects, stop
      if (objectNumber > 7)
      {
        break;
      }

      // get the centroid
      cv::Point2d centroid = cv::Point2d(centroids.at<double>(label, 0), centroids.at<double>(label, 1));

      // get the area
      double area = stats.at<int>(label, cv::CC_STAT_AREA);

      // create a binary mask
      cv::Mat mask = (labels == label);

      // calculate the central moments for the mask,
      // which are the moments defined relative to the centroid
      cv::Moments moments = cv::moments(mask, true);

      // get the angle of the axis of least central moment
      // mu11 is the second order central moment
      // mu20 and mu02 are the second order central moments along the x and y axes
      double alpha = 0.5 * atan2(2 * moments.mu11, moments.mu20 - moments.mu02);

      // plot the axis of least central moment
      draw_axis_of_least_central_moment(centroid, alpha, imageWithFeatures);

      // calculate the oriented bounding box corners and size
      std::vector<cv::Point2d> corners;
      std::vector<double> size;
      calculate_oriented_bounding_box(mask, centroid, alpha, corners, size);

      // calculate percent filled
      double percentFilled = area / (size[0] * size[1]);
      // calculate the height/width ratio
      double ratio = size[0] / size[1];

      // set the feature map
      featureMap[objectNumber] = {static_cast<float>(percentFilled), static_cast<float>(ratio)};

      // get the object label
      std::string objectLabel;
      if (k == 1)
      {
        objectLabel = get_object_label_nearest_neighbor(dbLabels, dbFeatures, percentFilled, ratio);
      }
      else
      {
        std::string objectLabel = get_object_label_k_nearest_neighbor(dbLabels, dbFeatures, percentFilled, ratio);
      }

      // plot the oriented bounding box and features
      draw_oriented_bounding_box(corners, percentFilled, ratio, objectNumber, objectLabel, imageWithFeatures);

      // increment the object number
      objectNumber++;
    }

    // display the image with features
    cv::imshow("With features", imageWithFeatures);

    // wait for a keypress
    int key = cv::waitKey(1);
    // if key is 'q', exit the loop and quit the program
    if (key == 'q')
    {
      break;
    }
    // if key is 's', save the frame to a file
    else if (key == 's')
    {
      // get filename and save the image
      std::string filename = get_image_name("../resources/images/", "original");
      cv::imwrite(filename, original);
      filename = get_image_name("../resources/images/", "binary");
      cv::imwrite(filename, binaryImage);
      filename = get_image_name("../resources/images/", "cleaned");
      cv::imwrite(filename, cleanedImage);
      filename = get_image_name("../resources/images/", "segmented");
      cv::imwrite(filename, segmentedImage);
      filename = get_image_name("../resources/images/", "with_features");
      cv::imwrite(filename, imageWithFeatures);
    }
    // if key is 'n', prompt user to input the object number and label
    else if (key == 'n')
    {
      // prompt the user for the object number and label
      std::string input = get_label_from_zenity();

      // parse the input
      std::string objectNumberString = input.substr(0, input.find("|"));
      std::string objectLabel = input.substr(input.find("|") + 1);

      // error checking
      if (objectNumberString == "" || objectLabel == "")
      {
        printf("error: invalid input\n");
        continue;
      }

      int objectNumber = std::stoi(objectNumberString);

      // write the data to a csv file
      append_object_data_csv("../resources/data.csv", objectLabel, featureMap[objectNumber]);

      // update the database
      dbLabels.push_back(objectLabel);
      dbFeatures.push_back(featureMap[objectNumber]);
    }
  }

  // free the video capture object
  delete vidCap;

  return (0);
}
