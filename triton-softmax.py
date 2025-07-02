import triton
import triton.language as tl
import torch

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

    # Compute pointers to the input and output row
    input_row = input_ptr + row_id * row_stride + col_offsets
    output_row = output_ptr + row_id * row_stride + col_offsets

    # Load input data with masking for columns beyond n_cols
    x = tl.load(input_row, mask=col_offsets < n_cols, other=-float('inf'))

    # Compute max for numerical stability
    x_max = tl.max(x, axis=0)

    # Subtract max and exponentiate
    x_exp = tl.exp(x - x_max)

    # Sum of exponentials
    x_sum = tl.sum(x_exp, axis=0)

    # Normalize to get softmax output
    softmax = x_exp / x_sum

    # Store output with mask
    tl.store(output_row, softmax, mask=col_offsets < n_cols)


def main():
    # Input dimensions
    rows, cols = 1024, 32

    # Create random input tensor on GPU
    input_tensor = torch.rand((rows, cols), dtype=torch.float32, device='cuda')

    # Prepare output tensor
    output_tensor = torch.empty_like(input_tensor)

    # Grid size = number of rows
    grid = (rows,)

    # Launch kernel
    softmax_kernel[grid](
        input_tensor, output_tensor,
        n_cols=cols,
        BLOCK_SIZE=cols,
        row_stride=cols
    )

    # Synchronize device
    torch.cuda.synchronize()

    # Print sample output
    print("Sample softmax output (first 2 rows):")
    print(output_tensor[:2])


if __name__ == "__main__":
    main()