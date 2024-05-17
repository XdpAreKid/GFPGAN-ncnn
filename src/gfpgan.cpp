// gfpgan implemented with ncnn library

#include "gfpgan.h"

GFPGAN::GFPGAN() {
  net.opt.use_vulkan_compute = false;
  net.opt.num_threads = 4;
}

GFPGAN::~GFPGAN() { net.clear(); }

int GFPGAN::load(const std::string &param_path, const std::string &model_path) {
  int ret = net.load_param(param_path.c_str());
  if (ret < 0) {
    fprintf(stderr, "open param file %s failed\n", param_path.c_str());
    return -1;
  }
  ret = net.load_model(model_path.c_str());
  if (ret < 0) {
    fprintf(stderr, "open bin file %s failed\n", model_path.c_str());
    return -1;
  }

  return 0;
}
int GFPGAN::process(const cv::Mat &img, ncnn::Mat &outimage) {
  ncnn::Mat ncnn_in = ncnn::Mat::from_pixels_resize(
      img.data, ncnn::Mat::PIXEL_BGR2RGB, img.cols, img.rows, 512, 512);
  ncnn_in.substract_mean_normalize(mean_vals, norm_vals);

  ncnn::Extractor ex = net.create_extractor();
  ex.input("in0", ncnn_in);

  ncnn::Mat ncnn_out;
  ex.extract("out0", ncnn_out);

  outimage = ncnn_out;

  return 0;
}
