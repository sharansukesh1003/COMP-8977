import torch
import triton
import triton.language as tl
import time
import nvtx  # Added for profiling

@triton.jit
def sigmoid_kernel(X_ptr, Y_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    x = tl.load(X_ptr + offsets, mask=mask)
    y = 1.0 / (1.0 + tl.exp(-x))
    tl.store(Y_ptr + offsets, y, mask=mask)

def sigmoid_triton(x):
    x = x.contiguous()
    y = torch.empty_like(x)
    n = x.numel()
    sigmoid_kernel[(triton.cdiv(n, 1024),)](x, y, n, BLOCK_SIZE=1024)
    torch.cuda.synchronize()
    return y

def main():
    x = torch.randn(1_000_000, device='cuda')

    torch.cuda.synchronize()
    start = time.time()
    for _ in range(999):
        with nvtx.annotate("triton_sigmoid", color="blue"):
            sigmoid_triton(x)
    y = sigmoid_triton(x)
    torch.cuda.synchronize()
    end = time.time()

    print("Avg time per run (ms):", (end - start) * 1000 / 1000)
    print("Sample Sigmoid output:", y[:10])

    # y = sigmoid_triton(x)
    # print("Sample Sigmoid output:", y[:10])

if __name__ == "__main__":
    main()

# Sample input and output
# x = torch.randn(4096, device='cuda')
# y = sigmoid_triton(x)
# print("Input tensor (first 10):", x[:10])
# print("Sigmoid output (first 10):", y[:10])
