/*
Copyright 2016 Fixstars Corporation

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http ://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#include <iostream>
#include <chrono>
#include <string>
#include <vector> // 用于存储每次的执行时间
#include <numeric> // 用于 accumulate
#include <iomanip>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp> // 用于 imread
#include <opencv2/highgui.hpp>   // 用于 CommandLineParser

#include <libsgm.h>

#include "sample_common.h"

// 命令行参数保持不变，以便指定输入图片
static const std::string keys =
    "{ @left-image-format  | <none> | format string for path to input left image   }"
    "{ @right-image-format | <none> | format string for path to input right image  }"
    "{ output_path         | .      | (unused) path to output directory            }"
    "{ disp_size           | 128    | maximum possible disparity value             }"
    "{ start_number        | 0      | index of the image pair to test              }"
    "{ total_number        | 0      | (unused) number of image pairs to process    }"
    "{ help h              |        | display this help and exit                   }";

std::string type2str(int type);

int main(int argc, char* argv[])
{
    cv::CommandLineParser parser(argc, argv, keys);
    if (parser.has("help")) {
        parser.printMessage();
        return 0;
    }

    const std::string image_format_L = parser.get<cv::String>("@left-image-format");
    const std::string image_format_R = parser.get<cv::String>("@right-image-format");
    const int disp_size = parser.get<int>("disp_size");
    const int start_number = parser.get<int>("start_number"); // 使用 start_number 来选择要测试的图片

    if (!parser.check()) {
        parser.printErrors();
        parser.printMessage();
        std::exit(EXIT_FAILURE);
    }

    // --- 修改部分开始: 将图片读取移至循环外 ---

    // 只读取由 start_number 指定的第一对（也是唯一一对）图片
    cv::Mat I1 = cv::imread(cv::format(image_format_L.c_str(), start_number), cv::IMREAD_UNCHANGED);
    cv::Mat I2 = cv::imread(cv::format(image_format_R.c_str(), start_number), cv::IMREAD_UNCHANGED);

    ASSERT_MSG(!I1.empty() && !I2.empty(), "imread failed. Check start_number and image paths.");
    ASSERT_MSG(I1.size() == I2.size() && I1.type() == I2.type(), "input images must be same size and type.");
    if (I1.type() != CV_8U && I1.type() != CV_16U) {
        std::cerr << "Error: Input image format is not supported." << std::endl;
        std::cerr << "Required format: CV_8U (8-bit grayscale) or CV_16U (16-bit grayscale)." << std::endl;
        std::cerr << "Actual format: " << type2str(I1.type()) << std::endl;
        std::cerr << "Hint: If you are using color images, please convert them to grayscale first." << std::endl;
        std::exit(EXIT_FAILURE);
    }
    // ASSERT_MSG(I1.type() == CV_8U || I1.type() == CV_16U, "input image format must be CV_8U or CV_16U.");
    ASSERT_MSG(disp_size == 64 || disp_size == 128 || disp_size == 256, "disparity size must be 64, 128 or 256.");

    const int width = I1.cols;
    const int height = I1.rows;

    const int src_depth = I1.type() == CV_8U ? 8 : 16;
    const int dst_depth = 16; // SGM 输出深度固定为 16
    const int src_bytes = src_depth * width * height / 8;
    const int dst_bytes = dst_depth * width * height / 8;

    sgm::StereoSGM sgm(width, height, disp_size, src_depth, dst_depth, sgm::EXECUTE_INOUT_CUDA2CUDA);

    device_buffer d_I1(src_bytes), d_I2(src_bytes), d_disparity(dst_bytes);
    
    // 将图片数据上传到GPU，此操作在循环外执行一次即可
    d_I1.upload(I1.data);
    d_I2.upload(I2.data);
    
    // --- 修改部分结束 ---


    // --- 新增计时逻辑 ---

    const int WARMUP_RUNS = 20;
    const int MEASUREMENT_RUNS = 50;
    double total_duration_us = 0.0;

    std::cout << "Starting performance measurement..." << std::endl;
    std::cout << "Warm-up runs: " << WARMUP_RUNS << std::endl;
    std::cout << "Measurement runs: " << MEASUREMENT_RUNS << std::endl;

    for (int i = 0; i < WARMUP_RUNS + MEASUREMENT_RUNS; ++i) {
        
        const auto t1 = std::chrono::high_resolution_clock::now();

        sgm.execute(d_I1.data, d_I2.data, d_disparity.data);
        cudaDeviceSynchronize(); // 确保GPU计算完成后再停止计时

        const auto t2 = std::chrono::high_resolution_clock::now();

        // 只在预热结束后开始累计时间
        if (i >= WARMUP_RUNS) {
            const auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
            total_duration_us += duration;
        }
    }

    // 计算并打印平均时间
    const double average_time_us = total_duration_us / MEASUREMENT_RUNS;
    const double average_time_ms = average_time_us / 1000.0;

    std::cout << "\n--------------------------------------------------" << std::endl;
    std::cout << "Performance Results:" << std::endl;
    std::cout << "Average execution time over " << MEASUREMENT_RUNS << " runs: "
              << std::fixed << std::setprecision(2) << average_time_ms << " ms." << std::endl;
    std::cout << "--------------------------------------------------" << std::endl;

    return 0;
}


// 将 OpenCV 的 Mat.type() 整数转换为可读的字符串
std::string type2str(int type) {
    std::string r;

    uchar depth = type & CV_MAT_DEPTH_MASK;
    uchar chans = 1 + (type >> CV_CN_SHIFT);

    switch (depth) {
    case CV_8U:  r = "8U"; break;
    case CV_8S:  r = "8S"; break;
    case CV_16U: r = "16U"; break;
    case CV_16S: r = "16S"; break;
    case CV_32S: r = "32S"; break;
    case CV_32F: r = "32F"; break;
    case CV_64F: r = "64F"; break;
    default:     r = "User"; break;
    }

    r += "C";
    r += std::to_string(chans);

    return r;
}