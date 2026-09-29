# src/run_all_experiments.py
import time
import numpy as np
from sklearn.metrics import accuracy_score
from data_loader import load_cifar10_subset
from features import apply_pca
from knn_ncc import knn_sklearn, nearest_class_centroid
from svm_custom import OneVsRestSVM
from svm_sklearn import svm_linear, svm_rbf, svm_poly
from mlp_numpy import MLP_Hinge
from visualization import plot_confusion_matrix, show_examples


def run_all():
    print("Loading CIFAR-10 subset...")
    x_train, y_train, x_test, y_test, x_test_raw = load_cifar10_subset()

    print("Applying PCA...")
    x_train_pca, x_test_pca, pca = apply_pca(x_train, x_test, 0.90)

    class_names = ["airplane", "automobile", "bird", "cat", "dog"]
    results = {}


    # Baselines
    print("\nRunning 1-NN...")
    preds, t_train, t_test = knn_sklearn(x_train_pca, y_train, x_test_pca, k=1)
    results["1-NN"] = (accuracy_score(y_test, preds), t_train, t_test)

    # === Visualization for 1-NN ===
    plot_confusion_matrix(y_test, preds, class_names, title="1-NN Confusion Matrix",
                          save_path="results/cm_1nn.png")

    print("\nRunning 3-NN...")
    preds, t_train, t_test = knn_sklearn(x_train_pca, y_train, x_test_pca, k=3)
    results["3-NN"] = (accuracy_score(y_test, preds), t_train, t_test)

    # === Visualization for 3-NN ===
    plot_confusion_matrix(y_test, preds, class_names, title="3-NN Confusion Matrix",
                          save_path="results/cm_3nn.png")

    print("\nRunning NCC...")
    preds, t_train, t_test = nearest_class_centroid(x_train_pca, y_train, x_test_pca)
    results["NCC"] = (accuracy_score(y_test, preds), t_train, t_test)

    # === Visualization for NCC ===
    plot_confusion_matrix(y_test, preds, class_names, title="NCC Confusion Matrix",
                          save_path="results/cm_ncc.png")

    # Custom SVM
    print("\nRunning Custom Linear SVM (OvR)...")
    svm = OneVsRestSVM(lr=1e-4, C=1.0, epochs=3)
    svm.fit(x_train_pca, y_train)
    preds = svm.predict(x_test_pca)
    results["Custom Linear SVM"] = (accuracy_score(y_test, preds), None, None)

    # === Visualization for Custom Linear SVM (OvR) ===
    plot_confusion_matrix(y_test, preds, class_names, title="Custom Linear SVM (OvR) Confusion Matrix",
                          save_path="results/cm_custom_linear_svm.png")


    # sklearn SVM 
    print("\nRunning Linear SVM (sklearn)...")
    preds, t_train, t_test = svm_linear(x_train_pca, y_train, x_test_pca)
    results["Linear SVM"] = (accuracy_score(y_test, preds), t_train, t_test)

    # === Visualization for Linear SVM (sklearn) ===
    plot_confusion_matrix(y_test, preds, class_names, title="Linear SVM (sklearn) Confusion Matrix",
                          save_path="results/cm_linear_svm.png")

    print("\nRunning RBF SVM...")
    preds, t_train, t_test = svm_rbf(x_train_pca, y_train, x_test_pca)
    results["RBF SVM"] = (accuracy_score(y_test, preds), t_train, t_test)

    # === Visualization for RBF SVM ===
    plot_confusion_matrix(y_test, preds, class_names, title="RBF SVM Confusion Matrix",
                          save_path="results/cm_rbf_svm.png")
    show_examples(x_test_raw, y_test, preds, class_names, correct=True, n=10,
                  save_path="results/rbf_correct_predictions.png")
    show_examples(x_test_raw, y_test, preds, class_names, correct=False, n=10,
                  save_path="results/rbf_wrong_predictions.png")

    print("\nRunning Polynomial SVM...")
    preds, t_train, t_test = svm_poly(x_train_pca, y_train, x_test_pca, degree=3)
    results["Poly SVM"] = (accuracy_score(y_test, preds), t_train, t_test)

    # === Visualization for Polynomial SVM ===
    plot_confusion_matrix(y_test, preds, class_names, title="Polynomial SVM Confusion Matrix",
                          save_path="results/cm_poly_svm.png")
    


    # MLP

    print("\nRunning MLP (NumPy)...")
    input_dim = x_train_pca.shape[1]
    mlp = MLP_Hinge(input_dim, 128, len(np.unique(y_train)), lr=1e-3)
    mlp.fit(x_train_pca, y_train, epochs=10, batch_size=128)
    preds = mlp.predict(x_test_pca)
    results["MLP (hinge)"] = (accuracy_score(y_test, preds), None, None)

    # === Visualization for MLP ===
    plot_confusion_matrix(y_test, preds, class_names, title="MLP Confusion Matrix",
                          save_path="results/cm_mlp_hinge.png")

   
    # Print results
    print("\n=== RESULTS ===")
    for model, (acc, t_train, t_test) in results.items():
        print(f"{model:20s} | Acc={acc:.4f} | Train={t_train} | Test={t_test}")

    return results


if __name__ == "__main__":
    run_all()

