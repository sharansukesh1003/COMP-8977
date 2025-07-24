import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms


# ======= CHOOSE WHICH FUNCTION TO USE =======
# Import only one activation per training run
# from Cuda.cuda_relu import cuda_relu
# from Cuda.cuda_sigmoid import cuda_sigmoid
# from Cuda.cuda_softmax import cuda_softmax
# from Triton.triton_relu import triton_relu
from Triton.triton_sigmoid import triton_sigmoid
# from Triton.triton_softmax import triton_softmax

# === CHOOSE WHICH function you want to use ===
#  inner activation:
ACTIVATION_FN = triton_sigmoid


# CNN model definition
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
    
    # For Triton-softmax only.
    # Return raw logits without applying softmax
    # Reason: CrossEntropyLoss in PyTorch internally applies log-softmax,
    # so we must pass raw scores (logits) here to ensure correct gradients.
    # Triton softmax will be applied manually later during evaluation/inference only.
    # def forward(self, x):
    #     x = self.conv1(x)
    #     x = nn.functional.relu(x)
    #     x = self.conv2(x)
    #     x = nn.functional.relu(x)
    #     x = self.dropout1(x)
    #     x = nn.functional.max_pool2d(x, 2)
    #     x = torch.flatten(x, 1)
    #     x = self.fc1(x)
    #     x = nn.functional.relu(x)
    #     x = self.dropout2(x)
    #     x = self.fc2(x)
    #     return x

# Training and data setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('.', train=True, download=True, transform=transforms.ToTensor()),
    batch_size=64, shuffle=True
)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('.', train=False, transform=transforms.ToTensor()),
    batch_size=1000, shuffle=False
)

model = FlexibleCNN(activation_fn=ACTIVATION_FN).to(device)
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
            output = triton_sigmoid(output) # for triton-softmax else comment out.
            preds = output.argmax(dim=1)
            correct += (preds == target).sum().item()
            total += target.size(0)
    return correct / total

#  training loop
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

# Save the trained model
torch.save(model.state_dict(), "cnn_with_triton_sigmoid.pth")
print("Model training complete and weights saved. Ready for benchmarking with Nsight Systems.")
