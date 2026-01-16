#include <iostream>
#include "opencv2/core.hpp"
#include <opencv2/core/mat.hpp>
#include <opencv2/imgcodecs.hpp>

int main() {
    cv::Mat inputImg = cv::imread("image.png", cv::IMREAD_UNCHANGED);

    if (inputImg.empty()) {
        std::cerr << "Error: Image not found!" << std::endl;
        return -1;
    }

    cv::Mat outputImg = cv::Mat::zeros(inputImg.size(), inputImg.type());

    //Access raw bytes of the image
    auto inputPtr = inputImg.ptr();


    outputImg.data = inputPtr;
    cv::imwrite("output.png", outputImg);
}