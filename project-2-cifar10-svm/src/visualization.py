# src/visualization.py
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
import os


def _save_and_show(save_path):
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=100)
    plt.show()

def plot_confusion_matrix(y_true, y_pred, class_names, title="Confusion Matrix", save_path=None):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    _save_and_show(save_path)


def show_examples(x_test_raw, y_true, y_pred, class_names, correct=True, n=10, save_path=None):
    if correct:
        idxs = np.where(y_true == y_pred)[0]
    else:
        idxs = np.where(y_true != y_pred)[0]

    if len(idxs) == 0:
        print("No examples found.")
        return

    idxs = np.random.choice(idxs, size=min(n, len(idxs)), replace=False)

    plt.figure(figsize=(12,3))
    for i, idx in enumerate(idxs):
        img = x_test_raw[idx]
        plt.subplot(1, n, i+1)
        plt.imshow(img)
        plt.axis("off")
        plt.title(f"T:{class_names[y_true[idx]]}\nP:{class_names[y_pred[idx]]}")

    plt.tight_layout()
    _save_and_show(save_path)
