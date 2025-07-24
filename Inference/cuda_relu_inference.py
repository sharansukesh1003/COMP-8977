import torch
import torch.nn as nn
from torchvision import datasets, transforms
import sys
import os 

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
_ = torch.ones(1).to(device)  
print(f"Using device: {device}")
import pycuda.autoinit  
from Cuda.cuda_relu import cuda_relu

ACTIVATION_FN = cuda_relu

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
        x = self.activation_fn(x)
        return x  

model = FlexibleCNN(activation_fn=ACTIVATION_FN).to(device)
model.load_state_dict(torch.load("cnn_with_cuda_relu.pth"))
model.eval()

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('.', train=False, transform=transforms.ToTensor()),
    batch_size=64, shuffle=False
)

with torch.no_grad():
    for data, _ in test_loader:
        data = data.to(device)
        logits = model(data)
        preds = logits.argmax(dim=1)
        print("Predictions:", preds.cpu().numpy())
        break  
