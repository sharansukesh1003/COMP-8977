import numpy as np
import os
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import time
import nvtx  # Added for Nsight profiling

# use the GCC 11 compiler installed via conda
gcc_path = os.path.join(os.environ["CONDA_PREFIX"], "bin", "x86_64-conda-linux-gnu-cc")

relu_kernel = SourceModule("""
__global__ void relu(float *x, float *y, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        y[idx] = fmaxf(0.0f, x[idx]);
    }
}
""", options=["-ccbin", gcc_path])

def relu_pycuda(x_np):
    n = x_np.size
    x_gpu = cuda.mem_alloc(x_np.nbytes)
    y_gpu = cuda.mem_alloc(x_np.nbytes)

    cuda.memcpy_htod(x_gpu, x_np)

    func = relu_kernel.get_function("relu")
    func(x_gpu, y_gpu, np.int32(n), block=(256,1,1), grid=((n+255)//256,1,1))

    cuda.Context.synchronize()  
    y_np = np.empty_like(x_np)
    cuda.memcpy_dtoh(y_np, y_gpu)
    return y_np

def main():
    x_np = np.random.randn(1_000_000).astype(np.float32)

    cuda.Context.synchronize()
    start = time.time()
    for _ in range(999):
        with nvtx.annotate("cuda_relu", color="red"):
            relu_pycuda(x_np)
    y_np = relu_pycuda(x_np)  # 1000th run to capture output
    cuda.Context.synchronize()
    end = time.time()
    print("Avg time per run (ms):", (end - start) * 1000 / 1000)

    print("Sample ReLU output:", y_np[:10])
    # y_np = relu_pycuda(x_np)
    # print("Sample ReLU output:", y_np[:10])

if __name__ == "__main__":
    main()

# x_np = np.random.randn(4096).astype(np.float32)
# print("PyCUDA ReLU:", relu_pycuda(x_np))
