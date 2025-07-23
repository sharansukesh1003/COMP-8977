import torch
import triton
import triton.language as tl

@triton.jit
def sigmoid_kernel(X_ptr, Y_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    x = tl.load(X_ptr + offsets, mask=mask)
    y = 1.0 / (1.0 + tl.exp(-x))
    tl.store(Y_ptr + offsets, y, mask=mask)

def triton_sigmoid(x):
    x = x.contiguous()
    y = torch.empty_like(x)
    n = x.numel()
    sigmoid_kernel[(triton.cdiv(n, 1024),)](x, y, n, BLOCK_SIZE=1024)
    torch.cuda.synchronize()
    return y
