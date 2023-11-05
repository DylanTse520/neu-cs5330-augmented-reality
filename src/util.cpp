/*
  Yixiang Xie
  Fall 2023
  CS 5330
*/

#include <fstream>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include "util.hpp"

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

// print a matrix
// mat: the matrix to print
// return: 0 if successful, -1 if error
int print_mat(cv::Mat mat)
{
  if (mat.empty())
  {
    printf("[]\n");
  }

  // print the matrix in formatted way
  for (int i = 0; i < mat.rows; i++)
  {
    if (mat.rows > 1)
    {
      if (i == 0)
      {
        printf("[ ");
      }
      else
      {
        printf("  ");
      }
    }
    for (int j = 0; j < mat.cols; j++)
    {
      if (j == 0)
      {
        printf("[");
      }
      else
      {
        printf(" ");
      }
      if (j == mat.cols - 1)
      {
        printf("%.2f]", mat.at<double>(i, j));
      }
      else
      {
        printf("%.2f, ", mat.at<double>(i, j));
      }
    }
    if (mat.rows > 1)
    {
      if (i == mat.rows - 1)
      {
        printf("]\n");
      }
      else
      {
        printf(",\n");
      }
    }
    else
    {
      printf("\n");
    }
  }

  return (0);
}

// convert the camera matrix and distortion coefficients to a vector
// cameraMatrix: the camera matrix
// distCoeffs: the distortion coefficients
// vec: the vector to store the result
// return: 0 if successful, -1 if error
int mat_to_vector(cv::Mat cameraMatrix, cv::Mat distCoeffs, std::vector<double> &vec)
{
  // check if the matrices are empty
  if (cameraMatrix.empty() || distCoeffs.empty())
  {
    printf("error: camera matrix or distortion coefficients is empty.\n");
    return (-1);
  }

  // convert the camera matrix to a vector
  for (int i = 0; i < cameraMatrix.cols; i++)
  {
    for (int j = 0; j < cameraMatrix.rows; j++)
    {
      vec.push_back(cameraMatrix.at<double>(j, i));
    }
  }

  // convert the distortion coefficients to a vector
  for (int i = 0; i < distCoeffs.cols; i++)
  {
    vec.push_back(distCoeffs.at<double>(0, i));
  }

  return (0);
}

// convert the vector to camera matrix and distortion coefficients
// cameraMatrix: the camera matrix
// distCoeffs: the distortion coefficients
// vec: the vector to store the result
// return: 0 if successful, -1 if error
int vector_to_mat(std::vector<double> vec, cv::Mat &cameraMatrix, cv::Mat &distCoeffs)
{
  // check if the vector is empty
  if (vec.empty())
  {
    printf("error: vector is empty.\n");
    return (-1);
  }

  // init the camera matrix and distortion coefficients
  cameraMatrix = cv::Mat::zeros(3, 3, CV_64F);
  distCoeffs = cv::Mat::zeros(1, 14, CV_64F);

  // convert the vector to camera matrix
  for (int i = 0; i < cameraMatrix.cols; i++)
  {
    for (int j = 0; j < cameraMatrix.rows; j++)
    {
      cameraMatrix.at<double>(j, i) = vec[i * cameraMatrix.rows + j];
    }
  }

  // get the size of the camera matrix
  int cameraMatrixSize = cameraMatrix.rows * cameraMatrix.cols;

  // convert the vector to distortion coefficients
  for (int i = 0; i < distCoeffs.cols; i++)
  {
    distCoeffs.at<double>(0, i) = vec[cameraMatrixSize + i];
  }

  return (0);
}

// read the object data from a obj file
// for each line in the obj file, if the line is a vertex, it should be in the format:
//   v x y z
// if the line is a face, it should be in the format:
//   f v1 v2 v3
// filename: the filename of the obj file
// vertices: the vector to store the vertices
// faces: the vector to store the faces
// return: 0 if successful, -1 if error
int read_object_data(std::string filename, std::vector<cv::Point3f> &vertices, std::vector<std::vector<int>> &faces)
{
  // open the file
  std::ifstream file(filename);

  // error checking
  if (!file.is_open())
  {
    printf("error: unable to open file.\n");
    return (-1);
  }

  // read the file line by line
  std::string line;
  while (std::getline(file, line))
  {
    // check if the line is empty
    if (line.empty())
    {
      continue;
    }

    // get the first word of the line, which indicates the type of the line
    std::string type;
    std::stringstream ss(line);
    ss >> type;

    // check if the line is a vertex
    if (type == "v")
    {
      // get the x, y, z coordinates of the vertex, and reverse the y and z coordinates
      float x, y, z;
      ss >> x >> z >> y;

      // error checking
      if (ss.fail())
      {
        printf("error: invalid vertex.\n");
        return (-1);
      }

      // create a vertex that is at the center of the chessboard
      cv::Point3f vertex(x + 4, y - 2.5, z);

      // add the vertex to the vector
      vertices.push_back(vertex);
    }

    // check if the line is a face
    else if (type == "f")
    {
      // create a face
      std::vector<int> face;

      // get the three vertices of the face
      int v1, v2, v3;
      ss >> v1 >> v2 >> v3;

      // error checking
      if (ss.fail())
      {
        printf("error: invalid face.\n");
        return (-1);
      }

      // add the vertices to the face, and subtract 1 because the vertices are 1-indexed
      face.push_back(v1 - 1);
      face.push_back(v2 - 1);
      face.push_back(v3 - 1);

      // add the face to the vector
      faces.push_back(face);
    }

    // error checking
    else
    {
      printf("error: invalid line type.\n");
      return (-1);
    }
  }

  // close the file
  file.close();

  return (0);
}

// draw the four outside corners of the chessboard
// and the 3D axes at the origin of the chessboard on the frame
// cameraMatrix: the camera matrix
// distCoeffs: the distortion coefficients
// rvec: the rotation vector
// tvec: the translation vector
// frame: the frame to draw on
// return: 0 if successful, -1 if error
int draw_corners(cv::Mat cameraMatrix, cv::Mat distCoeffs, cv::Vec3d rvec, cv::Vec3d tvec, cv::Mat &frame)
{
  // check if the camera matrix, distortion coefficients, and frame are empty
  if (cameraMatrix.empty() || distCoeffs.empty() || frame.empty())
  {
    printf("error: camera matrix, distortion coefficients, or frame is empty.\n");
    return (-1);
  }

  // define some 3D points in the space
  std::vector<cv::Point3f> objectPoints;
  // the four outside corners of the chessboard
  objectPoints.push_back(cv::Point3f(0, 0, 0));
  objectPoints.push_back(cv::Point3f(8, 0, 0));
  objectPoints.push_back(cv::Point3f(8, -5, 0));
  objectPoints.push_back(cv::Point3f(0, -5, 0));
  // the 3D axes at the origin of the chessboard
  objectPoints.push_back(cv::Point3f(2, 0, 0));
  objectPoints.push_back(cv::Point3f(0, -2, 0));
  objectPoints.push_back(cv::Point3f(0, 0, 2));
  // the 3D points of the rectangle masking the chessboard
  objectPoints.push_back(cv::Point3f(-1, 1, 0));
  objectPoints.push_back(cv::Point3f(9, 1, 0));
  objectPoints.push_back(cv::Point3f(9, -6, 0));
  objectPoints.push_back(cv::Point3f(-1, -6, 0));

  // project the 3D points to the image plane
  std::vector<cv::Point2f> imagePoints;
  cv::projectPoints(objectPoints, rvec, tvec, cameraMatrix, distCoeffs, imagePoints);

  // draw a rectangle masking the chessboard
  std::vector<cv::Point> squareCorners;
  squareCorners.push_back(imagePoints[7]);
  squareCorners.push_back(imagePoints[8]);
  squareCorners.push_back(imagePoints[9]);
  squareCorners.push_back(imagePoints[10]);
  std::vector<std::vector<cv::Point>> contours;
  contours.push_back(squareCorners);
  cv::polylines(frame, contours, true, cv::Scalar(255, 255, 255), 2);
  cv::fillPoly(frame, contours, cv::Scalar(255, 255, 255));

  // draw the four outside corners of the chessboard as circles
  cv::circle(frame, imagePoints[0], 6, cv::Scalar(0, 0, 0), -1);
  cv::circle(frame, imagePoints[1], 6, cv::Scalar(0, 0, 0), -1);
  cv::circle(frame, imagePoints[2], 6, cv::Scalar(0, 0, 0), -1);
  cv::circle(frame, imagePoints[3], 6, cv::Scalar(0, 0, 0), -1);

  // draw the 3D axes at the origin of the chessboard
  cv::line(frame, imagePoints[0], imagePoints[4], cv::Scalar(0, 0, 255), 3);
  cv::line(frame, imagePoints[0], imagePoints[5], cv::Scalar(0, 255, 0), 3);
  cv::line(frame, imagePoints[0], imagePoints[6], cv::Scalar(255, 0, 0), 3);

  return (0);
}

// draw the object on the frame
// cameraMatrix: the camera matrix
// distCoeffs: the distortion coefficients
// rvec: the rotation vector
// tvec: the translation vector
// vertices: the vertices of the object
// faces: the faces of the object
// frame: the frame to draw on
// return: 0 if successful, -1 if error
int draw_object(cv::Mat cameraMatrix, cv::Mat distCoeffs, cv::Vec3d rvec, cv::Vec3d tvec, std::vector<cv::Point3f> vertices, std::vector<std::vector<int>> faces, cv::Mat &frame)
{
  // check if the camera matrix, distortion coefficients, vertices, faces, and frame are empty
  if (cameraMatrix.empty() || distCoeffs.empty() || vertices.empty() || faces.empty() || frame.empty())
  {
    printf("error: camera matrix, distortion coefficients, vertices, faces, or frame is empty.\n");
    return (-1);
  }

  // project the vertices to the image plane
  std::vector<cv::Point2f> imagePoints;
  cv::projectPoints(vertices, rvec, tvec, cameraMatrix, distCoeffs, imagePoints);

  // draw the faces of the object
  for (int i = 0; i < faces.size(); i++)
  {
    // map the z coordinate of the face to a color
    float z = (vertices[faces[i][0]].z + vertices[faces[i][1]].z + vertices[faces[i][2]].z) / 3;
    int color = (int)(z / 3.5 * 155) + 100;
    cv::Scalar faceColor(color, color, color);

    // draw the face
    cv::line(frame, imagePoints[faces[i][0]], imagePoints[faces[i][1]], faceColor, 3);
    cv::line(frame, imagePoints[faces[i][1]], imagePoints[faces[i][2]], faceColor, 3);
    cv::line(frame, imagePoints[faces[i][2]], imagePoints[faces[i][0]], faceColor, 3);
  }

  return (0);
}