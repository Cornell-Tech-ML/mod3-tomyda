import minitorch
import time
import numpy as np
import matplotlib.pyplot as plt

FastTensorBackend = minitorch.TensorBackend(minitorch.FastOps)
GPUBackend = minitorch.TensorBackend(minitorch.CudaOps)


def run_matmul(backend: minitorch.TensorBackend, size: int = 16) -> None:
    """Perform matrix multiplication using the specified backend.

    Args:
        backend: The backend to use for matrix operations.
        size (int, optional): The size of the matrices. Defaults to 16.

    """
    batch_size = 2

    x = minitorch.rand((batch_size, size, size), backend=backend)
    y = minitorch.rand((batch_size, size, size), backend=backend)
    z = x @ y  # noqa: F841

def plot_results(results):
    sizes = list(results.keys())
    fast_times = [results[size]['fast'] for size in sizes]
    gpu_times = [results[size]['gpu'] for size in sizes]

    plt.figure(figsize=(10, 6))
    plt.plot(sizes, fast_times, label='Fast', marker='o')
    plt.plot(sizes, gpu_times, label='GPU', marker='o')
    plt.xlabel('Matrix Size')
    plt.ylabel('Time (seconds)')
    plt.title('Matrix Multiplication Performance')
    plt.legend()
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    # Warmup
    run_matmul(FastTensorBackend)
    run_matmul(GPUBackend)

    ntrials = 3
    times = {}
    for size in [16, 32, 64, 128, 256, 512, 1024, 2048, 4096]:
        print(f"Running size {size}")
        times[size] = {}
        simple_times = []
        fast_times = []
        gpu_times = []
        for _ in range(ntrials):
            start_fast = time.time()
            run_matmul(FastTensorBackend, size)
            end_fast = time.time()

            start_gpu = time.time()
            run_matmul(GPUBackend, size)
            end_gpu = time.time()

            fast_time = end_fast - start_fast
            gpu_time = end_gpu - start_gpu

            fast_times.append(fast_time)
            gpu_times.append(gpu_time)

        times[size]["fast"] = np.mean(fast_times)
        times[size]["gpu"] = np.mean(gpu_times)
        print(times[size])

    print()
    print("Timing summary")
    for size, stimes in times.items():
        print(f"Size: {size}")
        for b, t in stimes.items():
            print(f"    {b}: {t:.5f}")

    # Assuming `results` is the dictionary returned by `run_timing_tests`
    results = run_timing_tests(backend)
    plot_results(results)
