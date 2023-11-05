/*
  Yixiang Xie
  Fall 2023
  CS 5330
*/

#include <opencv2/opencv.hpp>
#include <vector>
#include "util.hpp"
#include "csv_util.h"

int main(int argc, char *argv[])
{
  // read the calibration result from a csv file
  std::vector<std::string> labels;
  std::vector<std::vector<double>> features;
  read_object_data_csv("../resources/data.csv", labels, features);

  // convert the calibration result to matrices
  cv::Mat cameraMatrix;
  cv::Mat distCoeffs;
  vector_to_mat(features[0], cameraMatrix, distCoeffs);

  // read the object data from a obj file
  std::vector<cv::Point3f> vertices;
  std::vector<std::vector<int>> faces;
  read_object_data("../resources/teapot.obj", vertices, faces);

  // the number of corners in the chessboard
  int cornersPerRow = 9;
  int cornersPerCol = 6;

  // create a vector of 3D points
  std::vector<cv::Vec3f> pointSet;
  for (int i = 0; i < cornersPerCol; i++)
  {
    for (int j = 0; j < cornersPerRow; j++)
    {
      pointSet.push_back(cv::Vec3f(j, -i, 0));
    }
  }

  // whether to use the camera or process a static image
  bool useCamera;
  cv::Mat image;
  cv::VideoCapture *vidCap;

  // read image path from command line
  if (argc > 1)
  {
    useCamera = false;

    // read the image
    image = cv::imread(argv[1]);
  }
  else
  {
    useCamera = true;

    // open the video device
    vidCap = new cv::VideoCapture(0);

    // error checking
    if (!vidCap->isOpened())
    {
      std::cerr << "error: unable to open video device" << std::endl;
      return (-2);
    }
  }

  cv::Mat frame;
  while (true)
  {
    // if using the camera
    if (useCamera)
    {
      // read a frame from the video stream
      *vidCap >> frame;
    }
    else // if using a static image
    {
      // copy the image to the frame
      frame = image.clone();
    }

    // error checking
    if (frame.empty())
    {
      std::cerr << "error: frame is empty" << std::endl;
      break;
    }

    // convert the frame to grayscale
    cv::Mat gray;
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

    // find the chessboard corners
    cv::Size pattern_size = cv::Size(cornersPerRow, cornersPerCol);
    std::vector<cv::Point2f> cornerSet;
    cv::TermCriteria termCrit(cv::TermCriteria::COUNT | cv::TermCriteria::EPS, 30, 0.01);
    bool found = cv::findChessboardCorners(gray, pattern_size, cornerSet);

    // if the corners are found
    if (found)
    {
      // refine the corner locations
      cv::cornerSubPix(gray, cornerSet, cv::Size(5, 5), cv::Size(-1, -1), termCrit);

      // calculate the pose of the chessboard
      cv::Vec3d rvec, tvec;
      cv::solvePnP(pointSet, cornerSet, cameraMatrix, distCoeffs, rvec, tvec);

      // print the pose of the chessboard
      std::cout << "rvec: " << rvec << std::endl;
      std::cout << "tvec: " << tvec << std::endl;

      // draw the four outside corners of the chessboard as circles
      // and the 3D axes at the origin of the chessboard
      draw_corners(cameraMatrix, distCoeffs, rvec, tvec, frame);

      // draw the object on the frame
      draw_object(cameraMatrix, distCoeffs, rvec, tvec, vertices, faces, frame);
    }

    // display the frame
    cv::imshow("AR", frame);

    // wait for a keypress
    int key = cv::waitKey(1);
    // if key is 'q', exit the loop and quit the program
    if (key == 'q')
    {
      break;
    }
    // if key is 's', save the frame
    else if (key == 's')
    {
      std::string filename = get_image_name("../resources/", "ar");
      cv::imwrite(filename, frame);
    }
  }

  // free the video capture object
  delete vidCap;

  return (0);
}
