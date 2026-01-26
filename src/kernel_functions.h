//
// Created by carlo on 19/01/26.
//

#ifndef SECOND_ASSIGNMENT_KERNEL_FUNCTIONS_H
#define SECOND_ASSIGNMENT_KERNEL_FUNCTIONS_H

#include <string>
#include <iostream>
#include <filesystem>
#include <fstream>

#define MAX_K 7
#define MAX_W 1024
#define MAX_H 1024
#define MAX_C 3

enum KernelType {
    Gaussian,
    Ridge,
    Sharpen,
    Identity,
    Gaussian7
};

struct Config {

    std::string datasetPath;
    int threads = 4;
    KernelType kernelType = Gaussian;
    std::string outputPath;
    int K = 3;
    bool first = true;


    //Parsing params passed via python
    void parse(int argc, char* argv[]) {
        for (int i = 1; i < argc; ++i) { //i = 1 because for i=0 we always have the exe path
            std::string arg = argv[i];
            if (arg == "--d" && i + 1 < argc) {
                datasetPath = argv[++i];
            }else if (arg == "--threads" && i + 1 < argc) {
                threads = std::stoi(argv[++i]);
            } else if (arg == "--type" && i + 1 < argc) {
                std::string stringType = argv[++i];
               if (stringType == "Gaussian") {
                   kernelType = Gaussian;
               }else if (stringType == "Ridge") {
                   kernelType = Ridge;
               }else if (stringType == "Sharpen") {
                   kernelType = Sharpen;
               }else if (stringType == "Identity") {
                   kernelType = Identity;
               }else if (stringType == "Gaussian7") {
                   kernelType = Gaussian7;
                   K = 7;
               }
            }else if (arg == "--output" && i + 1 < argc) {
                outputPath = argv[++i];
            }
            else {
                std::cerr << "Unknown argument: " << arg << std::endl;
            }
        }
    }
};



inline void generateKernel(float *kernel, KernelType type) {
    switch (type) {
        case Gaussian:
            for (int i = 0; i < 9; i++) {
                kernel[i] = 1.f / 9.f;
            }
            break;
        case Ridge:
            for (int i = 0; i < 9; i++) {
                kernel[i] = -1;
            }
            kernel[4] = 8;
        case Sharpen:
            for (int i = 0; i < 9; i++) {
                kernel[i] = 0;
            }
            kernel[4] = 5;
            kernel[1] = -1;
            kernel[3] = -1;
            kernel[5] = -1;
            kernel[7] = -1;
            break;
        case Gaussian7:
            for (int i = 0; i < 49; i++) {
                kernel[i] = 1.f / 49.f;
            }
            break;
        case Identity:
        default:
            for (int i = 0; i < 9; i++) {
                kernel[i] = 0;
            }
            kernel[4] = 1;
    }
}

inline unsigned char clamping(float x) {
    if (x < 0.0f) x = 0.0f;
    if (x > 255.0f) x = 255.0f;
    x = std::roundf(x);
    return static_cast<unsigned char> (x);
}

inline void append_csv(Config cfg, int dim_images, int num_images,double time_k, double time_e2e, std::string filename)
{

    auto exist = std::filesystem::exists(filename);

    std::ofstream out(filename, std::ios::app);

    if (!exist) {
        out << "N_threads,Kernel_size,dim_images,num_images,time_k,time_e2e\n";
        cfg.first = false;
    }

    out << cfg.threads << ","
        << cfg.K << ","
        << dim_images << ","
        << num_images << ","
        << time_k << ","
        << time_e2e << "\n";
}

#endif //SECOND_ASSIGNMENT_KERNEL_FUNCTIONS_H