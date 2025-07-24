import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda.compiler import SourceModule

# Optional: use system gcc if conda one causes issues
# gcc_path = os.path.join(os.environ["CONDA_PREFIX"], "bin", "x86_64-conda-linux-gnu-cc")

sigmoid_kernel = SourceModule("""
__global__ void sigmoid(float *x, float *y, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        float val = x[idx];
        y[idx] = 1.0f / (1.0f + expf(-val));
    }
}
""")

def cuda_sigmoid(x_torch):
    x_np = x_torch.detach().contiguous().cpu().numpy().astype(np.float32).copy()
    n = x_np.size
    x_gpu = cuda.mem_alloc(x_np.nbytes)
    y_gpu = cuda.mem_alloc(x_np.nbytes)
    cuda.memcpy_htod(x_gpu, x_np)
    func = sigmoid_kernel.get_function("sigmoid")
    func(x_gpu, y_gpu, np.int32(n), block=(256, 1, 1), grid=((n + 255) // 256, 1, 1))
    cuda.Context.synchronize()
    y_np = np.empty_like(x_np)
    cuda.memcpy_dtoh(y_np, y_gpu)
    return x_torch.new_tensor(y_np).view_as(x_torch)

# ========== CNN Definition ==========
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
        x = self.activation_fn(x)
        x = self.conv2(x)
        x = self.activation_fn(x)
        x = self.dropout1(x)
        x = nn.functional.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.activation_fn(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return x

# ========== Training Setup ==========
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('.', train=True, download=True, transform=transforms.ToTensor()),
    batch_size=64, shuffle=True
)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('.', train=False, transform=transforms.ToTensor()),
    batch_size=1000, shuffle=False
)

model = FlexibleCNN(activation_fn=cuda_sigmoid).to(device)
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
            preds = output.argmax(dim=1)
            correct += (preds == target).sum().item()
            total += target.size(0)
    return correct / total

# ========== Training Loop ==========
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

# ========== Save Model ==========
torch.save(model.state_dict(), "cnn_with_cuda_sigmoid.pth")
print("Model trained and saved using CUDA Sigmoid.")
