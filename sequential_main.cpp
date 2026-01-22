#include <iostream>
#include <string>
#include <filesystem>
#include <fstream>
namespace fs = std::filesystem;

#include "opencv2/core.hpp"
#include <opencv2/core/mat.hpp>
#include <opencv2/imgcodecs.hpp>
#include "src/parallel_kernel.h"
#include <omp.h>


int main() {

    omp_set_num_threads(4);

    auto start_e2e = std::chrono::high_resolution_clock::now();

    float kernel[9];
    generateKernel(kernel, Gaussian);

    std::string path = "../dataset_150x150/seg_pred/seg_pred";
    int count = 0;

    double total_k = 0;
    for (const auto & entry : fs::directory_iterator(path)) {
        cv::Mat inputImg = cv::imread(entry.path(), cv::IMREAD_UNCHANGED);
        cv::Mat outputImg = cv::Mat::zeros(inputImg.size(), inputImg.type());

        if (inputImg.empty()) {
            continue;
        }

        cv::Size size = inputImg.size();

        //Access raw bytes of the image
        auto inputPtr = inputImg.ptr();
        auto outputPtr = outputImg.ptr();

        auto start_k = std::chrono::high_resolution_clock::now();

        applyKernel<3>(inputPtr, outputPtr, kernel, size.width, size.height, inputImg.channels());

        auto end_k = std::chrono::high_resolution_clock::now();
        total_k += std::chrono::duration_cast<std::chrono::duration<double>>(end_k - start_k).count();

        count++;

        std::string outputPath = "../parallel_output/" + entry.path().filename().string();
        cv::imwrite(outputPath, outputImg);
    }
    auto end_e2e = std::chrono::high_resolution_clock::now();
    auto time_e2e = std::chrono::duration_cast<std::chrono::duration<double>>(end_e2e - start_e2e).count();

    std::cout << "Time Taken End to End:" << time_e2e << "s" << std::endl;
    std::cout << "Time Taken to apply kernel:" << total_k/count << "s" << std::endl;
    std::cout << "Elements Elaborated:" << count << std::endl;
}
