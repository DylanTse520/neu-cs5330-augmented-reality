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
  // open the video device
  cv::VideoCapture *vidCap;
  vidCap = new cv::VideoCapture(0);

  // error checking
  if (!vidCap->isOpened())
  {
    std::cerr << "error: unable to open video device" << std::endl;
    return (-2);
  }

  // get the width and height of frames in the video stream
  cv::Size refS((int)vidCap->get(cv::CAP_PROP_FRAME_WIDTH),
                (int)vidCap->get(cv::CAP_PROP_FRAME_HEIGHT));

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

  // the list of corner locations and 3D points
  std::vector<std::vector<cv::Point2f>> cornerList;
  std::vector<std::vector<cv::Vec3f>> pointList;

  // for all frames
  cv::Mat frame;
  while (true)
  {
    // read a frame from the video stream
    *vidCap >> frame;

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

      // draw the corners on the frame
      cv::drawChessboardCorners(frame, pattern_size, cornerSet, found);
    }

    // display the frame
    cv::imshow("Calibrate", frame);

    // wait for a keypress
    int key = cv::waitKey(1);
    // if key is 'q', exit the loop and quit the program
    if (key == 'q')
    {
      break;
    }
    // if key is 's', store the corner locations and 3D points, and save the image
    else if (key == 's')
    {
      // error checking
      if (!found)
      {
        std::cerr << "error: corners not found" << std::endl;
        continue;
      }

      // store the corner locations and 3D points
      cornerList.push_back(cornerSet);
      pointList.push_back(pointSet);

      // get filename and save the image
      std::string filename = get_image_name("../resources/", "calibrate");
      cv::imwrite(filename, frame);

      // if the number of images is no less than 5, calibrate the camera
      if (cornerList.size() >= 5)
      {
        // init the matrices
        cv::Mat cameraMatrix = cv::Mat::eye(3, 3, CV_64FC1);
        cameraMatrix.at<double>(0, 2) = refS.width / 2;
        cameraMatrix.at<double>(1, 2) = refS.height / 2;
        cv::Mat distCoeffs = cv::Mat::zeros(1, 14, CV_64FC1);

        // print the number of images
        printf("number of images used: %d\n", (int)cornerList.size());

        // print the initial matrices
        printf("initial camera matrix:\n");
        print_mat(cameraMatrix);
        printf("initial distortion coefficients:\n");
        print_mat(distCoeffs);

        // calibrate the camera assuming the pixel aspect ratio is 1 and radial distortion exists
        std::vector<cv::Mat> rvecs, tvecs;
        double reProjectionError = cv::calibrateCamera(pointList, cornerList, refS, cameraMatrix, distCoeffs, rvecs, tvecs, cv::CALIB_FIX_ASPECT_RATIO | cv::CALIB_RATIONAL_MODEL, termCrit);

        // print the calibrated matrices
        printf("calibrated camera matrix:\n");
        print_mat(cameraMatrix);
        printf("calibrated distortion coefficients:\n");
        print_mat(distCoeffs);

        // print the reprojection error
        printf("reprojection error: %f\n\n", reProjectionError);

        // store them in a vector
        std::vector<double> calibration;
        mat_to_vector(cameraMatrix, distCoeffs, calibration);

        // save the calibration result to a csv file
        append_object_data_csv("../resources/data.csv", "calibration", calibration, true);
      }
      else
      {
        printf("need at least 5 images for calibration. currently %d.\n", (int)cornerList.size());
      }
    }
  }

  // free the video capture object
  delete vidCap;

  return (0);
}
