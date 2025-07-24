import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

softmax_kernel = SourceModule("""
__global__ void softmax(float *x, float *y, int n) {
    float max_val = -INFINITY;
    for (int i = 0; i < n; i++) {
        if (x[i] > max_val) max_val = x[i];
    }

    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        sum += expf(x[i] - max_val);
    }

    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        y[idx] = expf(x[idx] - max_val) / sum;
    }
}
""")

def cuda_softmax(x_torch):
    x_np = x_torch.detach().contiguous().cpu().numpy().astype(np.float32)
    original_shape = x_np.shape
    x_flat = x_np.reshape(-1)
    n = x_flat.size

    import pycuda.gpuarray as gpuarray
    x_gpu = gpuarray.to_gpu(x_flat)
    y_gpu = gpuarray.empty_like(x_gpu)

    func = softmax_kernel.get_function("softmax")
    func(x_gpu, y_gpu, np.int32(n), block=(256,1,1), grid=((n+255)//256,1,1))
    # cuda.Context.synchronize() # gpuarray handles synchronization

    y_np = y_gpu.get()
    return x_torch.new_tensor(y_np).view(*original_shape)

# =================== CNN Model ===================
class FlexibleCNN(nn.Module):
    def __init__(self, activation_fn):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)
        self.activation_fn = activation_fn

    def forward(self, x):
      x = self.conv1(x)
      x = nn.functional.relu(x)
      x = self.conv2(x)
      x = nn.functional.relu(x)
      x = self.dropout1(x)
      x = nn.functional.max_pool2d(x, 2)
      x = torch.flatten(x, 1)
      x = self.fc1(x)
      x = nn.functional.relu(x)
      x = self.dropout2(x)
      x = self.fc2(x)  # raw logits
      return x

# =================== Training Setup ===================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('.', train=True, download=True, transform=transforms.ToTensor()),
    batch_size=64, shuffle=True
)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('.', train=False, transform=transforms.ToTensor()),
    batch_size=1000, shuffle=False
)

model = FlexibleCNN(activation_fn=cuda_softmax).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

def test_accuracy(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            output = cuda_softmax(output)
            preds = output.argmax(dim=1)
            correct += (preds == target).sum().item()
            total += target.size(0)
    return correct / total

# =================== Training Loop ===================
for epoch in range(10):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
    acc = test_accuracy(model, test_loader, device)
    print(f"Epoch {epoch+1}: accuracy={acc*100:.2f}% | Last loss: {loss.item():.4f}")

# =================== Save Model ===================
torch.save(model.state_dict(), "cnn_with_cuda_softmax.pth")
print("Model training complete and weights saved using CUDA softmax.")
