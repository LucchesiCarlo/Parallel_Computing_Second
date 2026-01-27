import os
import subprocess


#experiments settings

EXECUTABLES = {
    "CudaParallelBatch": "../cmake-build-release/CudaParallelBatch",
}


dataset_path = "../dataset_1024x1024"
Threads_values = [4]
N_experiments = 1
kernel_type = "Gaussian"

CSV_OUT = "CudaParallelBatch_k=3_1024x1024.csv"

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
    if os.path.exists(CSV_OUT):
        os.remove(CSV_OUT)
    for layout, exe in EXECUTABLES.items():
        for n_threads in Threads_values:
            for run_id in range(N_experiments):
                run_benchmarks(exe, dataset_path, n_threads, kernel_type, CSV_OUT)

if __name__ == "__main__":
    main()