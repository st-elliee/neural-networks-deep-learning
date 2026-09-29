# src/data_loader.py
import numpy as np
from torchvision.datasets import CIFAR10
from torchvision import transforms

# airplane=0, automobile=1, bird=2, cat=3, dog=5
SELECTED_CLASSES = [0, 1, 2, 3, 5]

def load_cifar10_subset():
    # convert images to tensors in [0, 1]
    transform = transforms.ToTensor()

    trainset = CIFAR10(root="./data", train=True, download=True, transform=transform)
    testset = CIFAR10(root="./data", train=False, download=True, transform=transform)

    x_train, y_train = [], []
    x_test, y_test = [], []

    # load training set (selected classes only)
    for img, label in trainset:
        if label in SELECTED_CLASSES:
            # img: [C,H,W] → [H,W,C]
            img_np = img.numpy().transpose(1, 2, 0)
            x_train.append(img_np)
            y_train.append(label)

    # load test set (selected classes only)
    for img, label in testset:
        if label in SELECTED_CLASSES:
            img_np = img.numpy().transpose(1, 2, 0)
            x_test.append(img_np)
            y_test.append(label)

    # convert to numpy arrays
    x_train = np.array(x_train, dtype=np.float32)
    x_test = np.array(x_test, dtype=np.float32)

    x_test_raw = x_test.copy()   # keep raw images for visualization

    # flatten: 32x32x3 → 3072
    x_train = x_train.reshape(len(x_train), -1)
    x_test = x_test.reshape(len(x_test), -1)

    # map labels to 0..4
    class_to_new = {c: i for i, c in enumerate(SELECTED_CLASSES)}
    y_train = np.array([class_to_new[c] for c in y_train])
    y_test = np.array([class_to_new[c] for c in y_test])

    return x_train, y_train, x_test, y_test, x_test_raw
