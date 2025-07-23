import numpy as np
import os
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray

# Get the correct GCC compiler path
gcc_path = os.path.join(os.environ["CONDA_PREFIX"], "bin", "x86_64-conda-linux-gnu-cc")

# CUDA kernel code for ReLU
kernel_code = """
#include <math.h>
__global__ void relu(float *x, float *y, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        y[idx] = fmaxf(0.0f, x[idx]);
    }
}
"""

# Compile kernel
try:
    relu_kernel = SourceModule(kernel_code, options=["--compiler-bindir=" + gcc_path])
    func = relu_kernel.get_function("relu")
except Exception as e:
    print("CUDA Kernel Compile Error:", e)
    exit(1)

# CUDA ReLU wrapper function
def cuda_relu(x_torch):
    x_np = x_torch.detach().contiguous().cpu().numpy().astype(np.float32)
    n = x_np.size
    x_gpu = gpuarray.to_gpu(x_np)
    y_gpu = gpuarray.empty_like(x_gpu)
    func(x_gpu, y_gpu, np.int32(n), block=(256,1,1), grid=((n+255)//256,1,1))
    y_np = y_gpu.get()
    return x_torch.new_tensor(y_np).view_as(x_torch)
