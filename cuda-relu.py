import pycuda.autoinit
import pycuda.driver as cuda
import numpy as np
from pycuda.compiler import SourceModule

relu_kernel = SourceModule("""
__global__ void relu(float *x, float *y, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        y[idx] = fmaxf(0.0f, x[idx]);
    }
}
""")

def relu_pycuda(x_np):
    n = x_np.size
    x_gpu = cuda.mem_alloc(x_np.nbytes)
    y_gpu = cuda.mem_alloc(x_np.nbytes)

    cuda.memcpy_htod(x_gpu, x_np)

    func = relu_kernel.get_function("relu")
    func(x_gpu, y_gpu, np.int32(n), block=(256,1,1), grid=((n+255)//256,1,1))

    y_np = np.empty_like(x_np)
    cuda.memcpy_dtoh(y_np, y_gpu)
    return y_np

x_np = np.random.randn(4096).astype(np.float32)
print("PyCUDA ReLU:", relu_pycuda(x_np))