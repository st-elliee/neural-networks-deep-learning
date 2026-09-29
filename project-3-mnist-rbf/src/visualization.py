# visualization.py
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def save_confusion_matrix(cm, title, outpath):
    ensure_dir(os.path.dirname(outpath))
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()

def save_examples(X_test, y_true, y_pred, title, outdir, n_show=20):
    """Save correct/wrong predictions as a text list and as image grids.
    X_test: raw (non-PCA) test images, shape [N, 784] or [N, 28, 28]."""
    ensure_dir(outdir)
    correct_idx = np.where(y_true == y_pred)[0][:n_show]
    wrong_idx   = np.where(y_true != y_pred)[0][:n_show]
    with open(os.path.join(outdir, "examples.txt"), "w") as f:
        f.write(f"{title}\n")
        f.write("Correct predictions (index, true -> pred):\n")
        for i in correct_idx:
            f.write(f"{i}: {y_true[i]} -> {y_pred[i]}\n")
        f.write("\nWrong predictions (index, true -> pred):\n")
        for i in wrong_idx:
            f.write(f"{i}: {y_true[i]} -> {y_pred[i]}\n")

    # image grids (first 10 of each)
    for name, idx in [("correct", correct_idx), ("wrong", wrong_idx)]:
        idx = idx[:10]
        if len(idx) == 0:
            continue
        fig, axes = plt.subplots(1, len(idx), figsize=(1.2 * len(idx), 1.9), squeeze=False)
        for ax, i in zip(axes[0], idx):
            ax.imshow(np.asarray(X_test[i]).reshape(28, 28), cmap="gray_r")
            ax.axis("off")
            ax.set_title(f"T:{y_true[i]}  P:{y_pred[i]}", fontsize=10,
                         color="black" if y_true[i] == y_pred[i] else "#c0392b")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"rbf_{name}_predictions.png"), dpi=110, bbox_inches="tight")
        plt.close()
