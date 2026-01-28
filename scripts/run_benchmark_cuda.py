import os
import subprocess


#experiments settings

EXECUTABLES = {
    "Cuda": "../cmake-build-release/Cuda",
    "CudaTile": "../cmake-build-release/CudaTile",
    "CudaBatch": "../cmake-build-release/CudaBatch",
    "CudaParallelBatch": "../cmake-build-release/CudaParallelBatch",
}


dataset_path = ["../dataset_150x150", "../dataset_1024x1024"]
Threads_values = [1,2,3,4]
N_experiments = 6
kernel_type = ["Gaussian", "Gaussian7"]



def run_benchmarks(exe, dataset_path, n_threads, kernel_type, output_path):

    env = os.environ.copy()

    cmd = [
        exe,
        "--d", dataset_path,
        "--threads", str(n_threads),
        "--type", kernel_type,
        "--output", output_path
    ]

    subprocess.run(cmd, env=env, text=True)

def main():

    for layout, exe in EXECUTABLES.items():
        csv = layout + ".csv"
        if os.path.exists(csv):
            os.remove(csv)
        for dataset in dataset_path:
            for kernel in kernel_type:
                for n_threads in Threads_values:
                    for _ in range(N_experiments):
                        run_benchmarks(exe, dataset, n_threads, kernel, csv)

if __name__ == "__main__":
    main()