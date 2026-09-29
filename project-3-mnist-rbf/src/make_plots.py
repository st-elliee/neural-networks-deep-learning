# make_plots.py
# Recreates the summary plots from the results recorded with train_rbf.py
import os
import matplotlib.pyplot as plt
import numpy as np

OUT = "../assets"
os.makedirs(OUT, exist_ok=True)


# 1. Accuracy vs K


K_values = [50, 100, 200, 400, 1500]
acc_K = [0.8846, 0.9183, 0.9387, 0.9530, 0.9698]

plt.figure(figsize=(7,5))
plt.plot(K_values, acc_K, marker='o', linewidth=2)
plt.title("RBF Accuracy vs Number of Hidden Neurons (K)")
plt.xlabel("K (Hidden Neurons)")
plt.ylabel("Accuracy")
plt.grid(True)
plt.savefig(f"{OUT}/plot_accuracy_vs_K.png", dpi=150)
plt.close()



# 2. Training Time vs K


time_K = [16.20, 28.33, 52.88, 97.94, 436.0]

plt.figure(figsize=(7,5))
plt.plot(K_values, time_K, marker='o', color='red', linewidth=2)
plt.title("Training Time vs Number of Hidden Neurons (K)")
plt.xlabel("K (Hidden Neurons)")
plt.ylabel("Training Time (seconds)")
plt.grid(True)
plt.savefig(f"{OUT}/plot_time_vs_K.png", dpi=150)
plt.close()



# 3. Accuracy vs PCA Variance


pca_var = [0.90, 0.95, 0.99]
acc_pca = [0.9400, 0.9387, 0.9387]

plt.figure(figsize=(7,5))
plt.plot(pca_var, acc_pca, marker='o', linewidth=2)
plt.title("RBF Accuracy vs PCA Variance")
plt.xlabel("PCA Variance Retained")
plt.ylabel("Accuracy")
plt.grid(True)
plt.savefig(f"{OUT}/plot_accuracy_vs_pca.png", dpi=150)
plt.close()


# 4. Bar chart: RBF vs NN vs NCC


models = ["NN", "NCC", "RBF (K=200)", "RBF (K=1500)"]
acc_models = [0.9694, 0.8199, 0.9387, 0.9698]

plt.figure(figsize=(7,5))
plt.bar(models, acc_models, color=["green", "orange", "blue", "purple"])
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")
plt.ylim(0.75, 1.0)
plt.grid(axis='y')
plt.savefig(f"{OUT}/plot_model_comparison.png", dpi=150)
plt.close()



# 5. Confusion Matrices (Combined Figure)

# Combine the saved confusion matrices into one figure

import matplotlib.image as mpimg

fig, axes = plt.subplots(1, 3, figsize=(15,5))

titles = ["NN Confusion", "NCC Confusion", "RBF Confusion (K=1500)"]
files = [
    f"{OUT}/cm_nn.png",
    f"{OUT}/cm_ncc.png",
    f"{OUT}/cm_rbf_K1500.png"
]


for ax, title, file in zip(axes, titles, files):
    img = mpimg.imread(file)
    ax.imshow(img)
    ax.set_title(title)
    ax.axis("off")

plt.tight_layout()
plt.savefig(f"{OUT}/plot_confusion_matrices_combined.png", dpi=150)
plt.close()

print("All plots saved to", OUT)
