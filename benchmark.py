
import torch
import time

def run_benchmark(device_name):
    """
    Performs a series of large matrix multiplications on a specified device. 
    
    Args:
        device_name (str): The device to run the benchmark on ('cpu' or 'cuda').
    
    Returns:
        float: The elapsed time in seconds.
    """
    if device_name == "cuda" and not torch.cuda.is_available():
        print("CUDA device not available. Skipping GPU benchmark.")
        return None

    device = torch.device(device_name)
    
    # Configuration for the benchmark
    matrix_size = 2048
    iterations = 500
    
    print(f"\nRunning benchmark on: {device_name.upper()}")
    print(f"Matrix Size: {matrix_size}x{matrix_size}")
    print(f"Iterations: {iterations}")

    # Create random matrices on the specified device
    a = torch.randn(matrix_size, matrix_size, device=device)
    b = torch.randn(matrix_size, matrix_size, device=device)

    # Warm-up run to handle any first-run overhead
    if device_name == "cuda":
        torch.cuda.synchronize()
    _ = torch.matmul(a, b)
    if device_name == "cuda":
        torch.cuda.synchronize()

    start_time = time.time()

    # Main benchmark loop
    for _ in range(iterations):
        torch.matmul(a, b)
    
    # Synchronize for accurate timing on GPU
    if device_name == "cuda":
        torch.cuda.synchronize()

    end_time = time.time()
    
    elapsed_time = end_time - start_time
    print(f"Elapsed Time: {elapsed_time:.4f} seconds")
    return elapsed_time

if __name__ == "__main__":
    print("Starting PyTorch Benchmark...")
    
    # --- GPU Benchmark ---
    gpu_time = run_benchmark("cuda")
    
    # --- CPU Benchmark ---
    cpu_time = run_benchmark("cpu")
    
    # --- Comparison ---
    if gpu_time and cpu_time and gpu_time > 0:
        speedup = cpu_time / gpu_time
        print(f"\n--- Results ---")
        print(f"GPU was {speedup:.2f} times faster than CPU.")
    else:
        print("\nCould not calculate speedup factor.")
