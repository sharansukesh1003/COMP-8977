import torch
import triton
import triton.language as tl

@triton.jit
def softmax_kernel(
    input_ptr, output_ptr,
    n_cols: tl.constexpr,
    row_stride: tl.constexpr
):
    row_id = tl.program_id(0)
    col_offsets = tl.arange(0, 128)  # Static BLOCK_SIZE
    input_row = input_ptr + row_id * row_stride + col_offsets
    output_row = output_ptr + row_id * row_stride + col_offsets
    x = tl.load(input_row, mask=col_offsets < n_cols, other=-float('inf'))
    x_max = tl.max(x, axis=0)
    x_exp = tl.exp(x - x_max)
    x_sum = tl.sum(x_exp, axis=0)
    softmax = x_exp / x_sum
    tl.store(output_row, softmax, mask=col_offsets < n_cols)

def triton_softmax(x):
    assert x.ndim == 2
    rows, cols = x.shape
    output = torch.empty_like(x)
    grid = (rows,)
    softmax_kernel[grid](
        x, output,
        n_cols=cols,
        row_stride=cols,
        num_warps=1,
        num_stages=1
    )
    torch.cuda.synchronize()
    return output
