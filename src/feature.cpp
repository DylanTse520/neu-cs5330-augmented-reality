/*
  Yixiang Xie
  Fall 2023
  CS 5330
*/

#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <vector>
#include "util.hpp"

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

  // for all frames
  cv::Mat frame;
  std::string featureType = "surf";
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

    if (featureType == "harris")
    {
      // find the harris corners
      cv::Mat dst = cv::Mat::zeros(refS.width, refS.height, CV_32FC1);
      cv::cornerHarris(gray, dst, 5, 3, 0.04);

      // normalize the result
      cv::Mat dstNorm, dstNormScaled;
      cv::normalize(dst, dstNorm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());
      cv::convertScaleAbs(dstNorm, dstNormScaled);

      // draw the corners
      for (int i = 0; i < dstNormScaled.rows; i++)
      {
        for (int j = 0; j < dstNormScaled.cols; j++)
        {
          if (dstNormScaled.at<uchar>(i, j) > 100)
          {
            cv::circle(frame, cv::Point(j, i), 3, cv::Scalar(0, 255, 255), 2);
          }
        }
      }
    }
    else if (featureType == "shi-tomasi")
    {
      // find the shi-tomasi corners
      std::vector<cv::Point2f> cornerSet;
      cv::goodFeaturesToTrack(gray, cornerSet, 100, 0.01, 10);

      // draw the corners
      for (int i = 0; i < cornerSet.size(); i++)
      {
        cv::circle(frame, cornerSet[i], 3, cv::Scalar(255, 0, 0), 2);
      }
    }
    else if (featureType == "sift")
    {
      // find the sift features
      cv::Ptr<cv::SIFT> sift = cv::SIFT::create();
      std::vector<cv::KeyPoint> keypoints;
      sift->detect(gray, keypoints);

      // draw the keypoints
      cv::drawKeypoints(frame, keypoints, frame, cv::Scalar(0, 255, 0));
    }
    else if (featureType == "surf")
    {
      // find the surf features
      cv::Ptr<cv::xfeatures2d::SURF> surf = cv::xfeatures2d::SURF::create();
      std::vector<cv::KeyPoint> keypoints;
      surf->detect(gray, keypoints);

      // draw the keypoints
      cv::drawKeypoints(frame, keypoints, frame, cv::Scalar(0, 0, 255));
    }

    // display the frame
    cv::imshow("Feature", frame);

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
      std::string filename = get_image_name("../resources/", featureType);
      cv::imwrite(filename, frame);
    }
    // if key is 'h', use harris corners
    else if (key == 'h')
    {
      featureType = "harris";
    }
    // if key is 't', use shi-tomasi corners
    else if (key == 't')
    {
      featureType = "shi-tomasi";
    }
    // if key is 'i', use sift features
    else if (key == 'i')
    {
      featureType = "sift";
    }
    // if key is 'u', use surf features
    else if (key == 'u')
    {
      featureType = "surf";
    }
  }

  // free the video capture object
  delete vidCap;

  return (0);
}
