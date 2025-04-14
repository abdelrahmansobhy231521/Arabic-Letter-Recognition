import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# === Device Setup ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === Data Paths ===
train_dir = r"D:\CodAlpha\Task3\archive (8)\Train Images 13440x32x32\train"
test_dir = r"D:\CodAlpha\Task3\archive (8)\Test Images 3360x32x32\test"

# === Data Loaders ===
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

train_dataset = datasets.ImageFolder(root=train_dir, transform=transform)
test_dataset = datasets.ImageFolder(root=test_dir, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

class_names = train_dataset.classes

# === CNN Model ===
class CNNModel(nn.Module):
    def __init__(self, num_classes):
        super(CNNModel, self).__init__()
        self.conv_block = nn.Sequential(
            self._conv_block(1, 16),   # Input layer
            self._conv_block(16, 34),
            self._conv_block(34, 64),
            self._conv_block(64, 128)
        )
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, num_classes)

    def _conv_block(self, in_channels, out_channels):
        conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        init.uniform_(conv.weight)  # Kernel initializer = uniform
        block = nn.Sequential(
            conv,
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),  # Activation = ReLU
            nn.MaxPool2d(2),
            nn.Dropout(0.2)
        )
        return block

    def forward(self, x):
        x = self.conv_block(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# === Initialize Model ===
num_classes = len(class_names)
model = CNNModel(num_classes).to(device)

# === Optimizer & Loss ===
optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

# === Training ===
def train(model, loader, optimizer, criterion, epochs=5):
    model.train()
    for epoch in range(epochs):
        total_loss, correct, total = 0, 0, 0
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        acc = 100 * correct / total
        print(f"Epoch [{epoch+1}/{epochs}] Loss: {total_loss:.4f} Accuracy: {acc:.2f}%")

# === Evaluation Metrics ===
def test_with_metrics(model, loader, class_names, save_dir="results"):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Classification report
    report = classification_report(all_labels, all_preds, target_names=class_names, digits=4)
    print("\n📊 Classification Report:\n", report)

    with open(os.path.join(save_dir, "classification_report.txt"), "w") as f:
        f.write(report)

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "confusion_matrix.png"))
    plt.close()
    print(f"✅ Report and confusion matrix saved to: {save_dir}")

# === Run Training & Testing ===
train(model, train_loader, optimizer, criterion, epochs=50)
test_with_metrics(model, test_loader, class_names)
