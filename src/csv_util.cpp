/*
Bruce A. Maxwell

CS 5330 Computer Vision
Spring 2021

modified by Yixiang Xie
Fall 2023

cpp functions for reading CSV files with a specific format
- first column is a string containing a object label
- every other column is a number

The function returns a std::vector of char* for the object labels and a 2D std::vector of doubles for the data
*/

#include <cstdio>
#include <cstring>
#include "opencv2/opencv.hpp"
#include "csv_util.h"

// reads a string from a CSV file. the 0-terminated string is returned in the char array os.
//
// The function returns false if it is successfully read. It returns true if it reaches the end of the line or the file.
int getstring(FILE *fp, char os[])
{
  int p = 0;
  int eol = 0;

  for (;;)
  {
    char ch = fgetc(fp);
    if (ch == ',')
    {
      break;
    }
    else if (ch == '\n' || ch == EOF)
    {
      eol = 1;
      break;
    }
    // printf("%c", ch ); // uncomment for debugging
    os[p] = ch;
    p++;
  }
  // printf("\n"); // uncomment for debugging
  os[p] = '\0';

  return (eol); // return true if eol
}

int getint(FILE *fp, int *v)
{
  char s[256];
  int p = 0;
  int eol = 0;

  for (;;)
  {
    char ch = fgetc(fp);
    if (ch == ',')
    {
      break;
    }
    else if (ch == '\n' || ch == EOF)
    {
      eol = 1;
      break;
    }

    s[p] = ch;
    p++;
  }
  s[p] = '\0'; // terminator
  *v = atoi(s);

  return (eol); // return true if eol
}

// utility function for reading one double value from a csv file
//
// fp: the file pointer
// v: the value
//
// returns true if it reaches the end of a line or the file
int getdouble(FILE *fp, double *v)
{
  char s[256];
  int p = 0;
  int eol = 0;

  for (;;)
  {
    char ch = fgetc(fp);
    if (ch == ',')
    {
      break;
    }
    else if (ch == '\n' || ch == EOF)
    {
      eol = 1;
      break;
    }

    s[p] = ch;
    p++;
  }
  s[p] = '\0'; // terminator
  *v = atof(s);

  return (eol); // return true if eol
}

int append_object_data_csv(std::string filename, std::string object_label, std::vector<double> &object_feature, int reset_file)
{
  char buffer[256];
  char mode[8];
  FILE *fp;

  strcpy(mode, "a");

  if (reset_file)
  {
    strcpy(mode, "w");
  }

  fp = fopen(filename.c_str(), mode);
  if (!fp)
  {
    printf("error: unable to open output file %s\n", filename.c_str());
    exit(-1);
  }

  // write the object label and the feature vector to the csv file
  strcpy(buffer, object_label.c_str());
  std::fwrite(buffer, sizeof(char), strlen(buffer), fp);
  for (int i = 0; i < object_feature.size(); i++)
  {
    char tmp[256];
    snprintf(tmp, 256, ",%f", object_feature[i]);
    std::fwrite(tmp, sizeof(char), strlen(tmp), fp);
  }

  std::fwrite("\n", sizeof(char), 1, fp); // EOL

  fclose(fp);

  return (0);
}

int read_object_data_csv(std::string filename, std::vector<std::string> &object_labels, std::vector<std::vector<double>> &object_features, int echo_file)
{
  FILE *fp;
  double fval;
  char object_label[256];

  fp = fopen(filename.c_str(), "r");
  if (!fp)
  {
    printf("error: unable to open object feature file\n");
    return (-1);
  }

  printf("reading %s\n", filename.c_str());
  for (;;)
  {
    std::vector<double> dvec;

    // read the object label
    if (getstring(fp, object_label))
    {
      break;
    }

    // read the whole feature file into memory
    for (;;)
    {
      // get next feature
      double eol = getdouble(fp, &fval);
      dvec.push_back(fval);
      if (eol)
        break;
    }

    object_features.push_back(dvec);

    char *fname = new char[strlen(object_label) + 1];
    strcpy(fname, object_label);
    object_labels.push_back(fname);
  }
  fclose(fp);
  printf("finished reading csv file\n");

  if (echo_file)
  {
    for (int i = 0; i < object_features.size(); i++)
    {
      printf("%s:  ", object_labels[i].c_str());
      for (int j = 0; j < object_features[i].size(); j++)
      {
        printf("%f  ", object_features[i][j]);
      }
      printf("\n");
    }
    printf("\n");
  }

  return (0);
}
