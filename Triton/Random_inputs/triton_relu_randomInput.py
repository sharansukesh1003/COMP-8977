import torch
import triton
import triton.language as tl
import time
import nvtx  # Added for profiling

@triton.jit
def relu_kernel(X_ptr, Y_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    x = tl.load(X_ptr + offsets, mask=mask)
    y = tl.maximum(x, 0.0)
    tl.store(Y_ptr + offsets, y, mask=mask)

def relu_triton(x):
    x = x.contiguous()
    y = torch.empty_like(x)
    n = x.numel()
    relu_kernel[(triton.cdiv(n, 1024),)](x, y, n, BLOCK_SIZE=1024)
    torch.cuda.synchronize()
    return y

def main():
    x = torch.randn(1_000_000, device='cuda')
    
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(999):
        with nvtx.annotate("triton_relu", color="green"):
            relu_triton(x)
    y = relu_triton(x)
    torch.cuda.synchronize()
    end = time.time()

    print("Avg time per run (ms):", (end - start) * 1000 / 1000)
    print("Sample ReLU output:", y[:10])

    # y = relu_triton(x)
    # print("Sample ReLU output:", y[:10])

if __name__ == "__main__":
    main()

# Example usage:
# x = torch.randn(4096, device='cuda')
# y = relu_triton(x)
# print("Input tensor (first 10):", x[:10])
# print("ReLU output tensor (first 10):", y[:10])
