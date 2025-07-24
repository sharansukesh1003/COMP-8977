import numpy as np
import os
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray

gcc_path = os.path.join(os.environ["CONDA_PREFIX"], "bin", "x86_64-conda-linux-gnu-cc")

kernel_code = """
__global__ void sigmoid(float *x, float *y, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        float val = x[idx];
        y[idx] = 1.0f / (1.0f + expf(-val));
    }
}
"""

try:
    sigmoid_kernel = SourceModule(kernel_code, options=["--compiler-bindir=" + gcc_path])
except Exception as e:
    print("CUDA Kernel Compile Error:", e)
    exit(1)

func = sigmoid_kernel.get_function("sigmoid")

def cuda_sigmoid(x_torch):
    x_np = x_torch.detach().contiguous().cpu().numpy().astype(np.float32)
    n = x_np.size

    x_gpu = gpuarray.to_gpu(x_np)
    y_gpu = gpuarray.empty_like(x_gpu)

    func(x_gpu, y_gpu, np.int32(n), block=(256, 1, 1), grid=((n + 255) // 256, 1, 1))
    cuda.Context.synchronize()

    y_np = y_gpu.get()
    return x_torch.new_tensor(y_np).view_as(x_torch)
