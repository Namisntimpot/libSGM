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
#include <sstream>
#include <iomanip>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp> // 用于 imread 和 imwrite
#include <opencv2/highgui.hpp>   // 用于 CommandLineParser

#include <libsgm.h>

#include "sample_common.h"

// 更新命令行参数，加入 output_path
static const std::string keys =
    "{ @left-image-format  | <none> | format string for path to input left image   }"
    "{ @right-image-format | <none> | format string for path to input right image  }"
    "{ output_path         | .      | path to output directory for disparity maps  }"
    "{ disp_size           | 128    | maximum possible disparity value             }"
    "{ start_number        | 0      | index to start reading                       }"
	"{ total_number        | 0      | number of image pairs to process             }"
    "{ help h              |        | display this help and exit                   }";

int main(int argc, char* argv[])
{
    cv::CommandLineParser parser(argc, argv, keys);
    if (parser.has("help")) {
        parser.printMessage();
        return 0;
    }

    const std::string image_format_L = parser.get<cv::String>("@left-image-format");
    const std::string image_format_R = parser.get<cv::String>("@right-image-format");
    const std::string output_path = parser.get<std::string>("output_path");
    const int disp_size = parser.get<int>("disp_size");
    const int start_number = parser.get<int>("start_number");
	const int total_number = parser.get<int>("total_number");

    if (!parser.check()) {
        parser.printErrors();
        parser.printMessage();
        std::exit(EXIT_FAILURE);
    }

    cv::Mat I1 = cv::imread(cv::format(image_format_L.c_str(), start_number), cv::IMREAD_UNCHANGED);
    cv::Mat I2 = cv::imread(cv::format(image_format_R.c_str(), start_number), cv::IMREAD_UNCHANGED);

    ASSERT_MSG(!I1.empty() && !I2.empty(), "imread failed. Check start_number and image paths.");
    ASSERT_MSG(I1.size() == I2.size() && I1.type() == I2.type(), "input images must be same size and type.");
    ASSERT_MSG(I1.type() == CV_8U || I1.type() == CV_16U, "input image format must be CV_8U or CV_16U.");
    ASSERT_MSG(disp_size == 64 || disp_size == 128 || disp_size == 256, "disparity size must be 64, 128 or 256.");

    const int width = I1.cols;
    const int height = I1.rows;

    const int src_depth = I1.type() == CV_8U ? 8 : 16;
    // 为了保证输出精度，将SGM的目标深度固定为16位
    const int dst_depth = 16;
    const int src_bytes = src_depth * width * height / 8;
    const int dst_bytes = dst_depth * width * height / 8;

    sgm::StereoSGM sgm(width, height, disp_size, src_depth, dst_depth, sgm::EXECUTE_INOUT_CUDA2CUDA);

    device_buffer d_I1(src_bytes), d_I2(src_bytes), d_disparity(dst_bytes);
    // 确保主机端的视差图为 CV_16S 类型，以接收SGM的输出
    cv::Mat disparity(height, width, CV_16S);

    const int invalid_disp = sgm.get_invalid_disparity();

    for (int frame_no = start_number; frame_no - start_number < total_number ; frame_no++) {

        I1 = cv::imread(cv::format(image_format_L.c_str(), frame_no), cv::IMREAD_UNCHANGED);
        I2 = cv::imread(cv::format(image_format_R.c_str(), frame_no), cv::IMREAD_UNCHANGED);
        if (I1.empty() || I2.empty()) {
            std::cout << "Finished processing all images or could not read image for frame " << frame_no << "." << std::endl;
            break; // 如果无法读取图像，则退出循环
        }

        d_I1.upload(I1.data);
        d_I2.upload(I2.data);

        const auto t1 = std::chrono::system_clock::now();

        sgm.execute(d_I1.data, d_I2.data, d_disparity.data);
        cudaDeviceSynchronize();

        const auto t2 = std::chrono::system_clock::now();
        const auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
        const double fps = 1e6 / duration;

        d_disparity.download(disparity.data);

        // --- 修改部分开始 ---

        // 创建一个16位无符号Mat用于保存
        cv::Mat output_disparity;
        // 将16位有符号的视差图(CV_16S)转换为16位无符号(CV_16U)，同时将所有像素值乘以100
        disparity.convertTo(output_disparity, CV_16U, 100.0);
        
        // 创建掩码来标记无效的视差值
        const cv::Mat mask = disparity == invalid_disp;
        // 将无效视差区域的像素值设置为0
        output_disparity.setTo(0, mask);
        
        // 格式化输出文件名，格式为 disparity_{i:04d}.png
        std::stringstream ss;
        ss << "disparity_" << std::setw(4) << std::setfill('0') << frame_no << ".png";
        std::string filename = ss.str();
        
        // 组合输出目录和文件名
        std::string full_path = output_path + "/" + filename;

        try {
            cv::imwrite(full_path, output_disparity);
            // 在控制台打印进度和性能信息
            std::cout << "Frame " << std::setw(4) << frame_no << ": Saved to " << full_path 
                      << " (" << std::fixed << std::setprecision(2) << fps << " FPS)" << std::endl;
        }
        catch (const cv::Exception& ex) {
            std::cerr << "Error saving frame " << frame_no << " to " << full_path << ". " << ex.what() << std::endl;
        }

        // --- 修改部分结束 ---
    }

    return 0;
}