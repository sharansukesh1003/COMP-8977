import torch

# Simple matrix multiplication on GPU
a = torch.randn((1024, 1024), device='cuda')
b = torch.randn((1024, 1024), device='cuda')

for _ in range(1000):
    c = torch.matmul(a, b)

torch.cuda.synchronize()
print("Test completed")
