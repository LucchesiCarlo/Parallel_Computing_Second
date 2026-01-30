# Kernel Image processing with OpenMP and CUDA

This is the repository for the second assignment of Parallel Computing, made by students Carlo Lucchesi and Giacomo
Orsucci.

## Objective

The goal of this project is to implement C/C++ code to apply convolutional kernel on images.
After a simple sequential implementation, we pivoted to implement different parallel versions and comparing them on some
test cases.

For **OpenMP**:

- Application of the kernel convolution.
- Parallel image loading.

For **CUDA**

- Kernel application using CUDA.
- Kernel convolution boosted with tiling.
- Applying the kernel in batches.
- Handling different batches in different `cudaStram`.

## Replicate experiments

For executing the testing in a systematic way, inside the directory `scripts` there are programs that we used:

- `run_benchmark_omp.py`: executes all OpenMP version on the 150x150 dataset.
- `run_benchmark_cuda.py`: executes all CUDA version on the 150x150 and 1024x1024 datasets.
- `on_big_dataset.sh`: executes only selected OpenMP version on the 1024x1024 dataset to confront with the CUDA
  versions.

**Attention:** remind to create the output directories for these processes, because OpenCV doesn't do it automatically
and so doesn't save anything.

For generating all the images of the report, we used the `Plot_Generator.ipynb` notebook jupyter.

### Results used

Also inside the `scripts` directory there are the `results.zip` that contains the `.csv` obtained during the test, and
used for the figures inside the report.

### Dataset Download

The dataset used for this project work can be easily downloaded using the `download_datasets.sh` script.
This scripts **doesn't** delete the zips files, so that can be extracted again if needed.

### OpenCV Requirements

For reading and writing files we used of OpenCV library, so it needs to be installed on the machine.
For Linux machines the command is:

```
sudo apt install libopencv-dev
```
