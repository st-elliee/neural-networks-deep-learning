# Neural Networks & Deep Learning

Coursework projects for the course *Neural Networks & Deep Learning*, Department of Electrical and Computer Engineering, Aristotle University of Thessaloniki (Fall 2025).

Each project trains a model from scratch in **Python**, runs a set of experiments on its hyperparameters, and compares it against classical baselines (**k-Nearest Neighbors**, **Nearest Class Centroid**).

## Projects

| # | Project | Method | Dataset | Best result |
|---|---|---|---|---|
| 1 | [CIFAR-10 classification with a CNN](project-1-cifar10-cnn/) | CNN (PyTorch) | CIFAR-10 | 75.95% test accuracy (vs. 38.73% for kNN) |
| 2 | [SVMs vs. baseline classifiers](project-2-cifar10-svm/) | SVM (linear / poly / RBF), custom SVM & MLP in NumPy | CIFAR-10 (5 classes) | 65.14% with RBF SVM (vs. 51.32% for 1-NN) |
| 3 | *coming soon* | | | |

Each folder contains the code, a README with results and figures, and the full project report (`docs/report.pdf`).

## Tech stack

Python · PyTorch · torchvision · scikit-learn · NumPy · Matplotlib · seaborn
