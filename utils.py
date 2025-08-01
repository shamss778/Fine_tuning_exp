import scipy.io as sio
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from PIL import Image
from torch import nn
from torchvision.datasets import SVHN


transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)), 
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

class SVHNDataset(Dataset):
    def __init__(self, mat_file, transform=None):
        self.data = sio.loadmat(mat_file)
        self.images = self.data['X'].transpose(3, 0, 1, 2)  # Transpose to (N, H, W, C)
        self.labels = self.data['y'].squeeze()  # Remove extra dimension
        self.labels[self.labels == 10] = 0  # Change label 10 to 0 for digit 0
        self.transform = transform

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        image = Image.fromarray(image)
        
        if self.transform:
            image = self.transform(image)
        
        return image, label
    

class LeNet5(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 6, 5),
            nn.Tanh(),
            nn.MaxPool2d(2),
            nn.Conv2d(6, 16, 5),
            nn.Tanh(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(16*4*4, 120),
            nn.Tanh(),
            nn.Linear(120, 84),
            nn.Tanh(),
            nn.Linear(84, 10) 
        )
    
    def forward(self, x):
        return self.net(x)
    
train_dataset = SVHNDataset('/Users/shamsbenmefteh/Documents/Fine_tuning_exp/data/SVHN/train_32x32.mat', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=6, shuffle=True)

test_dataset = SVHNDataset('/Users/shamsbenmefteh/Documents/Fine_tuning_exp/data/SVHN/test_32x32.mat', transform=transform)
test_loader = DataLoader(test_dataset, batch_size=6, shuffle=False)

def train(model, train_loader, criterion, optimizer, epochs):
    for epoch in range(epochs):  # 5 epochs as example
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels.long())
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}")

def test(model, test_loader, criterion):
    model.eval()
    test_loss, test_acc = 0.0, 0
    with torch.no_grad():
        for x, y in test_loader:
            preds = model(x)
            test_loss += criterion(preds, y).item()
            test_acc  += (preds.argmax(1)==y).sum().item()
    print(f"Test   : loss={test_loss/len(test_loader):.4f}, acc={test_acc/len(test_dataset):.4f}")