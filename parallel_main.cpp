//
// Created by giacomo on 17/01/26.
//

#include <iostream>
#include <string>
#include <filesystem>
#include <fstream>
namespace fs = std::filesystem;

#include "opencv2/core.hpp"
#include <opencv2/core/mat.hpp>
#include <opencv2/imgcodecs.hpp>
#include "src/sequential_kernel.h"
#include <omp.h>

int main() {

    omp_set_num_threads(4);

    auto start_e2e = std::chrono::high_resolution_clock::now();

    float kernel[49];
    generateKernel(kernel, Gaussian7);

    std::string path = "../dataset_150x150/seg_pred/seg_pred";


    double total_k = 0;
    std::vector<std::filesystem::path> imgList;

    for (const auto & entry : fs::directory_iterator(path)) {
        imgList.push_back(entry.path());
    }

#pragma omp parallel for default(none) shared(imgList, kernel, total_k)
    for (int i=0; i<imgList.size(); i++) {
        cv::Mat inputImg = cv::imread(imgList[i], cv::IMREAD_UNCHANGED);
        cv::Mat outputImg = cv::Mat::zeros(inputImg.size(), inputImg.type());

        if (inputImg.empty()) {
            continue;
        }

        cv::Size size = inputImg.size();

        //Access raw bytes of the image
        auto inputPtr = inputImg.ptr();
        auto outputPtr = outputImg.ptr();

        auto start_k = std::chrono::high_resolution_clock::now();
        applyKernel<7>(inputPtr, outputPtr, kernel, size.width, size.height, inputImg.channels());
        auto end_k = std::chrono::high_resolution_clock::now();

        auto temp = std::chrono::duration_cast<std::chrono::duration<double>>(end_k - start_k).count();


        std::string outputPath = "../parallel_image_output/" + imgList[i].filename().string();
        cv::imwrite(outputPath, outputImg);
#pragma omp atomic
        total_k += temp;

    }

    auto end_e2e = std::chrono::high_resolution_clock::now();
    auto time_e2e = std::chrono::duration_cast<std::chrono::duration<double>>(end_e2e - start_e2e).count();


    std::cout << "Time Taken End to End:" << time_e2e << "s" << std::endl;
    std::cout << "Time Taken to apply kernel:" << total_k/imgList.size() << "s" << std::endl;
    std::cout << "Elements Elaborated:" << imgList.size() << std::endl;

}
