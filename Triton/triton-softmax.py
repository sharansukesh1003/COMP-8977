import triton
import triton.language as tl
import torch
import time
import nvtx  # Added for profiling

# Triton kernel: row-wise softmax optimized for cols=32
@triton.jit
def softmax_kernel(
    input_ptr, output_ptr,
    n_cols: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    row_stride: tl.constexpr
):
    row_id = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_row = input_ptr + row_id * row_stride + col_offsets
    output_row = output_ptr + row_id * row_stride + col_offsets
    x = tl.load(input_row, mask=col_offsets < n_cols, other=-float('inf'))
    x_max = tl.max(x, axis=0)
    x_exp = tl.exp(x - x_max)
    x_sum = tl.sum(x_exp, axis=0)
    softmax = x_exp / x_sum
    tl.store(output_row, softmax, mask=col_offsets < n_cols)

def main():
    # Input dimensions
    rows, cols = 32768, 32
    input_tensor = torch.rand((rows, cols), dtype=torch.float32, device='cuda')
    output_tensor = torch.empty_like(input_tensor)
    grid = (rows,)

    torch.cuda.synchronize()
    start = time.time()
    for _ in range(999):
        with nvtx.annotate("triton_softmax", color="purple"):
            softmax_kernel[grid](
                input_tensor, output_tensor,
                n_cols=cols,
                BLOCK_SIZE=cols,
                row_stride=cols
            )
    softmax_kernel[grid](
        input_tensor, output_tensor,
        n_cols=cols,
        BLOCK_SIZE=cols,
        row_stride=cols
    )
    torch.cuda.synchronize()
    end = time.time()

    print("Avg time per run (ms):", (end - start) * 1000 / 1000)
    print("Sample softmax output (first 2 rows):")
    print(output_tensor[:2])

    # print("Sample softmax output (first 2 rows):")
    # print(output_tensor[:2])

if __name__ == "__main__":
    main()
