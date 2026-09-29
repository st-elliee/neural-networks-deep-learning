import torch
import torchvision
import torchvision.transforms as transforms
from   torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import time
import csv
import matplotlib.pyplot as plt
import os
import numpy as np
from   sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import torchvision.transforms.functional as F
import argparse

# Command-line arguments
parser = argparse.ArgumentParser(description="Train a small CNN on CIFAR-10")
parser.add_argument("--neurons", type=int, default=1024,
                    help="hidden units in the fully connected layer (e.g. 128, 256, 384, 512, 1024)")
parser.add_argument("--lr", type=float, default=0.0005, help="Adam learning rate")
parser.add_argument("--epochs", type=int, default=10, help="number of training epochs")
args = parser.parse_args()



# Select device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Image transforms (normalization to [-1, 1])
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),
                         (0.5, 0.5, 0.5))
])

# Load CIFAR-10 (downloaded automatically to ./data)
trainset = torchvision.datasets.CIFAR10(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

testset = torchvision.datasets.CIFAR10(
    root='./data',
    train=False,
    download=True,
    transform=transform
)

# DataLoaders
trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
testloader  = DataLoader(testset, batch_size=64, shuffle=False)

# Class labels
classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')


NEURONS = args.neurons

class CifarCNN(nn.Module):
    def __init__(self):
        super(CifarCNN, self).__init__()

        # Block 1
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1   = nn.BatchNorm2d(32)
        self.pool  = nn.MaxPool2d(2, 2)  # 32x32 -> 16x16

        # Block 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2   = nn.BatchNorm2d(64)
        # MaxPool 16x16 -> 8x8

        # Fully Connected
        self.fc1 = nn.Linear(64 * 8 * 8, NEURONS)
        self.fc2 = nn.Linear(NEURONS, 10)



    def forward(self, x):
        x = self.pool(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool(torch.relu(self.bn2(self.conv2(x))))
        x = x.view(-1, 64 * 8 * 8)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

net = CifarCNN().to(device)

# Loss + Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=args.lr)




# training

history = []

num_epochs = args.epochs
start_time = time.time()

print("\nStarting Training...\n")
for epoch in range(num_epochs):
    net.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0


    for i, data in enumerate(trainloader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)

        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()


        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{num_epochs} - Loss: {running_loss/len(trainloader):.4f}")

        # --- Test accuracy per epoch ---
    net.eval()
    correct_test = 0
    total_test = 0
    with torch.no_grad():
        for images, labels in testloader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total_test += labels.size(0)
            correct_test += (predicted == labels).sum().item()

    train_acc = 100 * correct_train / total_train
    test_acc = 100 * correct_test / total_test

    # save results for CSV
    history.append([
        epoch+1,
        running_loss / len(trainloader),
        train_acc,
        test_acc
    ])

    net.train()


training_time = time.time() - start_time
print("\nFinished Training!")
print(f"Training Time: {training_time:.2f} seconds\n")


# testing



net.eval()
correct = 0
total = 0
all_preds = []
all_labels = []

with torch.no_grad():
    for data in testloader:
        images, labels = data[0].to(device), data[1].to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

accuracy = 100 * correct / total
print(f"Accuracy on test set: {accuracy:.2f}%\n")


# confusion matrix
cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)


disp.plot(cmap=plt.cm.Blues, xticks_rotation='vertical')
plt.title(f"Confusion Matrix (Accuracy {accuracy:.2f}%)")
plt.tight_layout()


os.makedirs("results", exist_ok=True)
plt.savefig(f"results/confusion_matrix_{NEURONS}.png")
plt.show()

# --- Save Correct & Wrong Predictions ---


# Create folders
os.makedirs(f"results/correct_{NEURONS}", exist_ok=True)
os.makedirs(f"results/wrong_{NEURONS}", exist_ok=True)

correct_samples = []
wrong_samples = []

net.eval()
with torch.no_grad():
    for images, labels in testloader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = net(images)
        _, predicted = torch.max(outputs, 1)

        for i in range(images.size(0)):
            img = images[i].cpu()
            true_label = labels[i].item()
            pred_label = predicted[i].item()

            # undo normalization
            img = (img * 0.5) + 0.5
            img = F.to_pil_image(img)

            if true_label == pred_label:
                correct_samples.append((img, true_label, pred_label))
            else:
                wrong_samples.append((img, true_label, pred_label))

print(f"Correct: {len(correct_samples)}  |  Wrong: {len(wrong_samples)}")

# Save FIRST 10 correct and FIRST 10 wrong
def save_imgs(samples, folder, max_imgs=10):
    for i, (img, true_l, pred_l) in enumerate(samples[:max_imgs]):
        img.save(f"{folder}/{i:03d}_true-{classes[true_l]}_pred-{classes[pred_l]}.png")

save_imgs(correct_samples, f"results/correct_{NEURONS}")
save_imgs(wrong_samples,  f"results/wrong_{NEURONS}")

print("Saved correct & wrong prediction samples.")

# --- Show Samples in Output ---
def show_grid(samples, title, num=10):
    plt.figure(figsize=(12, 6))
    for i, (img, t, p) in enumerate(samples[:num]):
        plt.subplot(2, 5, i+1)
        plt.imshow(img)
        plt.axis("off")
        plt.title(f"T:{classes[t]}, P:{classes[p]}")
    plt.suptitle(title)
    plt.show()

show_grid(correct_samples, "Correct Predictions", num=10)
show_grid(wrong_samples,  "Wrong Predictions", num=10)


with open("results/training_log.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Epoch", "Train Loss", "Train Accuracy (%)", "Test Accuracy (%)"])
    writer.writerows(history)

print("Saved training log to results/training_log.csv")


# --- Extract values for plotting ---
epochs = [row[0] for row in history]
train_losses = [row[1] for row in history]
train_accuracies = [row[2] for row in history]
test_accuracies = [row[3] for row in history]



# 1) Plot Training Loss per Epoch

plt.figure()
plt.plot(epochs, train_losses, marker='o')
plt.xlabel("Epoch")
plt.ylabel("Training Loss")
plt.title("Training Loss per Epoch")
plt.grid(True)
plt.savefig(f"results/loss_{NEURONS}.png")
plt.show()



# 2) Plot Training vs Test Accuracy

plt.figure()
plt.plot(epochs, train_accuracies, label="Train Accuracy", marker='o')
plt.plot(epochs, test_accuracies, label="Test Accuracy", marker='o')
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Training & Test Accuracy per Epoch")
plt.legend()
plt.grid(True)
plt.savefig(f"results/accuracy_{NEURONS}.png")
plt.show()
