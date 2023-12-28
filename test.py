import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18
from pytorch_lightning import seed_everything

seed_everything(42)

# Step 1: Load pre-trained ResNet model
pretrained_resnet = resnet18(pretrained=True)
num_classes_cifar10 = 10  # CIFAR-10 has 10 classes

# Move the model to CUDA if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Step 2: Modify the last fully connected layer for CIFAR-10
pretrained_resnet.fc = nn.Linear(pretrained_resnet.fc.in_features, num_classes_cifar10)

if not os.path.exists('resnet18_cifar10-svd.pth'):
    from utils import convert_ckpt_to_svd
    state_dict = convert_ckpt_to_svd(pretrained_resnet)
    torch.save(state_dict, 'resnet18_cifar10-svd.pth')

pretrained_resnet = pretrained_resnet.to(device)

# Step 3: Test the model on CIFAR-10
def test(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    print(f'Test Accuracy on CIFAR-10: {accuracy * 100:.2f}%')

# CIFAR-10 dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

# Test the pre-trained model on CIFAR-10
test(pretrained_resnet, testloader)

# Step 4: Fine-tune the model on CIFAR-10 for a few epochs
num_epochs = 5
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(pretrained_resnet.parameters(), lr=0.001)

for epoch in range(num_epochs):
    pretrained_resnet.train()
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = pretrained_resnet(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f'Epoch {epoch + 1}, Loss: {running_loss / len(trainloader)}')

# Step 5: Test the accuracy after fine-tuning
test(pretrained_resnet, testloader)
