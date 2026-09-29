# train_rbf.py
import argparse
import time
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from data import load_mnist, fit_pca, apply_pca
from baselines import baseline_nn, baseline_ncc
from rbf import train_rbf_classifier, predict_rbf
from visualization import save_confusion_matrix, save_examples

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pca-variance", type=float, default=0.95)
    parser.add_argument("--centers-mode", type=str, default="kmeans", choices=["kmeans", "random"])
    parser.add_argument("--sigma-mode", type=str, default="median", choices=["median", "per_center"])
    parser.add_argument("--K", type=int, default=1500)
    parser.add_argument("--limit", type=int, default=None, help="Optional subset size for speed")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--outdir", type=str, default="./results")
    args = parser.parse_args()

    # Load
    (X_train, y_train), (X_test, y_test) = load_mnist(flatten=True, limit=args.limit, seed=args.seed)
    # PCA
    pca = fit_pca(X_train, variance=args.pca_variance, svd_solver="full", random_state=args.seed)
    Z_train, Z_test = apply_pca(pca, X_train, X_test)
    print(f"PCA components: {pca.n_components_}")

    # Baselines
    nn_res  = baseline_nn(Z_train, y_train, Z_test, y_test, k=1)
    ncc_res = baseline_ncc(Z_train, y_train, Z_test, y_test)
    print(f"Baseline NN acc={nn_res['acc']:.4f}, time={nn_res['time']:.2f}s")
    print(f"Baseline NCC acc={ncc_res['acc']:.4f}, time={ncc_res['time']:.2f}s")

    # Train RBF and evaluate on the official MNIST test set
    model = train_rbf_classifier(
        Z_train, y_train,
        Z_test, y_test,
        K=args.K,
        centers_mode=args.centers_mode,
        sigma_mode=args.sigma_mode,
        ridge=args.ridge,
        seed=args.seed
    )
    y_pred_test = predict_rbf(Z_test, model)
    acc_test = accuracy_score(y_test, y_pred_test)
    cm_test = confusion_matrix(y_test, y_pred_test)
    print(f"RBF K={args.K} centers={args.centers_mode} sigma={args.sigma_mode} acc={acc_test:.4f}, time_train={model['time_train']:.2f}s")

    # Save figures
    save_confusion_matrix(cm_test, title=f"RBF Confusion (K={args.K})", outpath=f"{args.outdir}/rbf_confusion_K{args.K}.png")
    save_confusion_matrix(nn_res["cm"], title="NN Confusion", outpath=f"{args.outdir}/nn_confusion.png")
    save_confusion_matrix(ncc_res["cm"], title="NCC Confusion", outpath=f"{args.outdir}/ncc_confusion.png")

    # Save examples of correct/incorrect
    save_examples(X_test, y_test, y_pred_test, title=f"RBF Examples K={args.K}", outdir=args.outdir)

    # Summary
    print("Summary:")
    print(f"- PCA components: {pca.n_components_}")
    print(f"- NN acc:  {nn_res['acc']:.4f}")
    print(f"- NCC acc: {ncc_res['acc']:.4f}")
    print(f"- RBF acc: {acc_test:.4f}")

if __name__ == "__main__":
    main()
