import os
import subprocess


#experiments settings

EXECUTABLES = {
    "CudaBatch": "../cmake-build-release/CudaBatch",
}


dataset_path = "../dataset_150x150"
Threads_values = [1]
N_experiments = 1
kernel_type = "Gaussian"

CSV_OUT = "CudaBatch_k=3_150x150.csv"

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