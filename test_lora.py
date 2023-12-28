import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18
from utils import register_low_rank, get_lora_delta_state_dict, load_lora_ckpt
from pytorch_lightning import seed_everything

seed_everything(42)

learning_rate = 0.01
weight1d_lr_multiplier = 1
weight2d_lr_multiplier = 1
weight3d_lr_multiplier = 1
weight4d_lr_multiplier = 1
bias_lr_multiplier = 0

learning_rates = {
        '1d': learning_rate * weight1d_lr_multiplier,
        '2d': learning_rate * weight2d_lr_multiplier,
        '3d': learning_rate * weight3d_lr_multiplier,
        '4d': learning_rate * weight4d_lr_multiplier,
        'bias': learning_rate * bias_lr_multiplier,
    }

num_classes_cifar10 = 10  # CIFAR-10 has 10 classes

# Step 1: Load pre-trained ResNet model
pretrained_resnet = resnet18(pretrained=True)

# Move the model to CUDA if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Step 2: Modify the last fully connected layer for CIFAR-10
pretrained_resnet.fc = nn.Linear(pretrained_resnet.fc.in_features, num_classes_cifar10)
pretrained_fc_weight = pretrained_resnet.fc.weight.data.clone()
pretrained_fc_bias = pretrained_resnet.fc.bias.data.clone()
pretrained_resnet = pretrained_resnet.to(device)

parametrized_module_list = []
new_module_dict = {'1d': [], '2d': [], '3d': [], '4d': [], 'bias': []}
new_module_params = {'1d': [], '2d': [], '3d': [], '4d': [], 'bias': []}
register_low_rank(
    pretrained_resnet,
    cached_svd_state_dict=torch.load('resnet18_cifar10-svd.pth'),
    bias_trainable=False,
    svd_kwargs={},
    parametrized_module_list=parametrized_module_list,
    new_module_dict=new_module_dict,
    new_module_params=new_module_params,
)
new_module_model = torch.nn.ModuleDict({k: torch.nn.ModuleList(v) for k, v in new_module_dict.items()})
params_to_optimize = {
    k: new_module_model[k].parameters() for k in new_module_model.keys() if (learning_rates[k] > 0)
}
print(new_module_dict)
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
optimizer = optim.Adam([
            {'params': params_to_optimize[k], 'lr': learning_rates[k]} for k in params_to_optimize.keys()])

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

test(pretrained_resnet, testloader)

state_dict, running = get_lora_delta_state_dict(pretrained_resnet)
state_dict = {'state_dict': state_dict, 'running': running}
torch.save(state_dict, 'resnet18_cifar10_lora.pth')

loaded_resnet = resnet18(pretrained=True)
loaded_resnet.fc = nn.Linear(loaded_resnet.fc.in_features, num_classes_cifar10)
with torch.no_grad():
    loaded_resnet.fc.weight.data.copy_(pretrained_fc_weight)
    loaded_resnet.fc.bias.data.copy_(pretrained_fc_bias)
loaded_resnet = loaded_resnet.to(device)
# test(loaded_resnet, testloader)

loaded_resnet = load_lora_ckpt(
    loaded_resnet,
    'resnet18_cifar10_lora.pth',
    'resnet18_cifar10-svd.pth',
    weight_kwargs={
        'svd_delta_scales_1d': [1.0],
        'svd_delta_scales_2d': [1.0],
        'svd_delta_scales_3d': [1.0],
        'svd_delta_scales_4d': [1.0],
    },
)
if len(state_dict['running']) > 0:
    loaded_resnet.load_state_dict(state_dict['running'], strict=False)

# Step 5: Test the accuracy after fine-tuning
test(loaded_resnet, testloader)