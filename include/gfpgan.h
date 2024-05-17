// gfpgan implemented with ncnn library

#ifndef GFPGAN_H
#define GFPGAN_H

#include <chrono>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <ostream>
#include <random>
#include <stdio.h>
#include <vector>
// ncnn
#include "cpu.h"
#include "layer.h"
#include "net.h"

class GFPGAN {
public:
  GFPGAN();
  ~GFPGAN();

  int load(const std::string &param_path, const std::string &model_path);

  int process(const cv::Mat &img, ncnn::Mat &outimage);

private:
  const float mean_vals[3] = {127.5f, 127.5f, 127.5f};
  const float norm_vals[3] = {1 / 127.5f, 1 / 127.5f, 1 / 127.5f};
  ncnn::Mat const_input;
  ncnn::Net net;
};

#endif // GFPGAN_H
