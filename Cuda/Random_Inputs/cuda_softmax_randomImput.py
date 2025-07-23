import numpy as np
import os
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import time
import nvtx  # Added

# use the GCC 11 compiler installed via conda
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
    if (tid < cols) val = input[idx];

    float row_max = warpReduceMax(val);
    row_max = __shfl_sync(0xffffffff, row_max, 0);

    float ex = expf(val - row_max);
    float sum_exp = warpReduceSum(ex);
    sum_exp = __shfl_sync(0xffffffff, sum_exp, 0);

    if (tid < cols)
        output[idx] = ex / sum_exp;
}
"""

mod = SourceModule(kernel_code, options=["-ccbin", gcc_path])
softmax_kernel = mod.get_function("softmax_kernel")

def main():
    rows, cols = 32768, 32
    input_data = np.random.rand(rows, cols).astype(np.float32)
    output_data = np.empty_like(input_data)

    d_input = cuda.mem_alloc(input_data.nbytes)
    d_output = cuda.mem_alloc(output_data.nbytes)
    cuda.memcpy_htod(d_input, input_data)

    block_size = (cols, 1, 1)
    grid_size = (rows, 1, 1)

    cuda.Context.synchronize()
    start = time.time()
    for _ in range(999):
        with nvtx.annotate("cuda_softmax", color="orange"):
            softmax_kernel(d_input, d_output, np.int32(cols), block=block_size, grid=grid_size)
    softmax_kernel(d_input, d_output, np.int32(cols), block=block_size, grid=grid_size)
    cuda.Context.synchronize()
    end = time.time()
    print("Avg time per run (ms):", (end - start) * 1000 / 1000)

    cuda.memcpy_dtoh(output_data, d_output)
    print("Sample softmax output (first 2 rows):")
    print(output_data[:2])

    # print("Sample softmax output (first 2 rows):")
    # print(output_data[:2])

if __name__ == "__main__":
    main()
