import numpy as np
import os
import pycuda.driver as cuda
import torch

# You can keep pycuda.autoinit import optional, it initializes context on import
import pycuda.autoinit  

# gcc path for options (keep if needed)
gcc_path = os.path.join(os.environ["CONDA_PREFIX"], "bin", "x86_64-conda-linux-gnu-cc")

kernel_code = """
#define WARP_SIZE 32
__inline__ __device__ float warpReduceMax(float val) {
    for (int offset = WARP_SIZE/2; offset > 0; offset /= 2)
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    return val;
}
__inline__ __device__ float warpReduceSum(float val) {
    for (int offset = WARP_SIZE/2; offset > 0; offset /= 2)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}
extern "C"
__global__ void softmax_kernel(const float* input, float* output, int cols) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int idx = row * cols + tid;

    float val = -INFINITY;
    if (tid < cols) {
        val = input[idx];
    }

    float row_max = warpReduceMax(val);
    row_max = __shfl_sync(0xffffffff, row_max, 0);

    float ex = 0.0f;
    if (tid < cols) {
        ex = expf(val - row_max);
    }

    float sum_exp = warpReduceSum(ex);
    sum_exp = __shfl_sync(0xffffffff, sum_exp, 0);

    if (tid < cols) {
        output[idx] = ex / sum_exp;
    }
}
"""

def cuda_softmax(x_torch):
    """
    x_torch: 2D torch tensor on CUDA device, shape (batch_size, num_classes)
    returns: softmax probabilities same shape and device/type as input
    """
    # Import here to ensure pycuda context is initialized AFTER torch.cuda is active
    from pycuda.compiler import SourceModule

    # Compile the kernel here to bind to current CUDA context (after torch usage)
    mod = SourceModule(kernel_code, options=["-ccbin", gcc_path])
    softmax_kernel = mod.get_function("softmax_kernel")

    x_np = x_torch.detach().cpu().numpy()
    if x_np.ndim != 2:
        x_np = x_np.reshape(x_np.shape[0], -1)

    rows, cols = x_np.shape
    output_np = np.empty_like(x_np)

    d_input = cuda.mem_alloc(x_np.nbytes)
    d_output = cuda.mem_alloc(output_np.nbytes)
    cuda.memcpy_htod(d_input, x_np)

    block_size = (32, 1, 1)  # Warp size of 32 threads per block for softmax
    grid_size = (rows, 1, 1)  # One block per row

    # Launch kernel
    softmax_kernel(d_input, d_output, np.int32(cols), block=block_size, grid=grid_size)
    cuda.Context.synchronize()

    cuda.memcpy_dtoh(output_np, d_output)

    return torch.from_numpy(output_np).to(x_torch.device).type_as(x_torch).view_as(x_torch)
