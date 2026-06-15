"""
Synthetic Dataset Generation Module
====================================
This module provides functions to generate various types of synthetic datasets
for machine learning experiments, including classification, regression, and
clustering tasks.

Author: Synthetic Data Generation Module
Version: 1.0
"""

import numpy as np
from math import sqrt, pi, sin, cos


def generate_linearly_separable_data(n_samples=200, n_features=2, random_seed=42):
    """
    Generate a linearly separable binary classification dataset.
    
    Creates two clusters that can be separated by a linear decision boundary.
    
    Parameters:
    -----------
    n_samples : int, optional
        Total number of samples to generate (default: 200)
    n_features : int, optional
        Number of features per sample (default: 2)
    random_seed : int, optional
        Random seed for reproducibility (default: 42)
    
    Returns:
    --------
    dict
        Dictionary containing:
        - "X": feature matrix (n_samples x n_features)
        - "y": binary labels (0 or 1)
        - "dataset_type": name of dataset
        - "n_samples": number of samples
        - "n_features": number of features
        - "n_classes": number of classes (2)
    """
    
    np.random.seed(random_seed)
    
    n_samples_per_class = n_samples // 2
    
    # Class 0: centered at origin with negative offset
    X_class_0 = np.random.randn(n_samples_per_class, n_features) - 2
    y_class_0 = np.zeros(n_samples_per_class, dtype=int)
    
    # Class 1: centered away from origin with positive offset
    X_class_1 = np.random.randn(n_samples_per_class, n_features) + 2
    y_class_1 = np.ones(n_samples_per_class, dtype=int)
    
    # Combine and shuffle
    X = np.vstack([X_class_0, X_class_1])
    y = np.hstack([y_class_0, y_class_1])
    
    # Shuffle
    shuffle_idx = np.random.permutation(n_samples)
    X = X[shuffle_idx]
    y = y[shuffle_idx]
    
    return {
        "X": X,
        "y": y,
        "dataset_type": "Linearly Separable",
        "n_samples": len(X),
        "n_features": n_features,
        "n_classes": 2
    }


def generate_xor_dataset(n_samples=200, noise=0.0, random_seed=42):
    """
    Generate XOR (non-linearly separable) binary classification dataset.
    
    Creates a dataset where the two classes are separated by an XOR pattern,
    which requires non-linear decision boundaries.
    
    Parameters:
    -----------
    n_samples : int, optional
        Total number of samples to generate (default: 200)
    noise : float, optional
        Standard deviation of Gaussian noise added to features (default: 0.0)
    random_seed : int, optional
        Random seed for reproducibility (default: 42)
    
    Returns:
    --------
    dict
        Dictionary containing:
        - "X": feature matrix (n_samples x 2)
        - "y": binary labels (0 or 1)
        - "dataset_type": name of dataset
        - "n_samples": number of samples
        - "n_features": number of features (2)
        - "n_classes": number of classes (2)
    """
    
    np.random.seed(random_seed)
    
    # Generate points uniformly in [0, 1] x [0, 1]
    X = np.random.uniform(0, 1, (n_samples, 2))
    
    # XOR labels: class 1 if (x1 XOR x2), else class 0
    y = ((X[:, 0] > 0.5) != (X[:, 1] > 0.5)).astype(int)
    
    # Add noise if specified
    if noise > 0:
        X += np.random.randn(n_samples, 2) * noise
    
    return {
        "X": X,
        "y": y,
        "dataset_type": "XOR (Non-linear)",
        "n_samples": len(X),
        "n_features": 2,
        "n_classes": 2
    }


def generate_multiclass_dataset(n_samples=300, n_features=3, n_classes=3, random_seed=42):
    """
    Generate a multi-class classification dataset.
    
    Creates multiple Gaussian clusters, one for each class.
    
    Parameters:
    -----------
    n_samples : int, optional
        Total number of samples to generate (default: 300)
    n_features : int, optional
        Number of features per sample (default: 3)
    n_classes : int, optional
        Number of classes to generate (default: 3)
    random_seed : int, optional
        Random seed for reproducibility (default: 42)
    
    Returns:
    --------
    dict
        Dictionary containing:
        - "X": feature matrix (n_samples x n_features)
        - "y": class labels (0 to n_classes-1)
        - "dataset_type": name of dataset
        - "n_samples": number of samples
        - "n_features": number of features
        - "n_classes": number of classes
    """
    
    np.random.seed(random_seed)
    
    n_samples_per_class = n_samples // n_classes
    X_list = []
    y_list = []
    
    # Generate clusters for each class
    for class_idx in range(n_classes):
        # Center of cluster
        center = np.random.randn(n_features) * 3
        
        # Generate samples around center
        X_class = np.random.randn(n_samples_per_class, n_features) + center
        y_class = np.full(n_samples_per_class, class_idx, dtype=int)
        
        X_list.append(X_class)
        y_list.append(y_class)
    
    # Combine and shuffle
    X = np.vstack(X_list)
    y = np.hstack(y_list)
    
    shuffle_idx = np.random.permutation(len(X))
    X = X[shuffle_idx]
    y = y[shuffle_idx]
    
    return {
        "X": X,
        "y": y,
        "dataset_type": "Multi-class Gaussian",
        "n_samples": len(X),
        "n_features": n_features,
        "n_classes": n_classes
    }


def generate_regression_dataset(n_samples=200, n_features=1, noise_level=0.1, random_seed=42):
    """
    Generate a regression dataset.
    
    Creates synthetic data for regression tasks with configurable noise.
    
    Parameters:
    -----------
    n_samples : int, optional
        Number of samples to generate (default: 200)
    n_features : int, optional
        Number of input features (default: 1)
    noise_level : float, optional
        Standard deviation of noise added to target (default: 0.1)
    random_seed : int, optional
        Random seed for reproducibility (default: 42)
    
    Returns:
    --------
    dict
        Dictionary containing:
        - "X": feature matrix (n_samples x n_features)
        - "y": continuous target values
        - "dataset_type": name of dataset
        - "n_samples": number of samples
        - "n_features": number of features
    """
    
    np.random.seed(random_seed)
    
    # Generate random features
    X = np.random.randn(n_samples, n_features)
    
    # Generate target: y = sum(X) + quadratic + noise
    if n_features == 1:
        y = 2 * X[:, 0] + X[:, 0]**2 - 1
    else:
        y = np.sum(X, axis=1) + np.sum(X**2, axis=1) * 0.5
    
    # Add noise
    y += np.random.randn(n_samples) * noise_level
    
    return {
        "X": X,
        "y": y,
        "dataset_type": "Regression (Polynomial)",
        "n_samples": len(X),
        "n_features": n_features
    }


def generate_imbalanced_dataset(n_samples=200, imbalance_ratio=0.1, random_seed=42):
    """
    Generate an imbalanced binary classification dataset.
    
    Creates a dataset where one class is significantly less frequent than the other,
    simulating real-world imbalanced classification problems.
    
    Parameters:
    -----------
    n_samples : int, optional
        Total number of samples to generate (default: 200)
    imbalance_ratio : float, optional
        Ratio of minority class to total samples (default: 0.1 = 10%)
    random_seed : int, optional
        Random seed for reproducibility (default: 42)
    
    Returns:
    --------
    dict
        Dictionary containing:
        - "X": feature matrix (n_samples x 2)
        - "y": binary labels (0 or 1) with class imbalance
        - "dataset_type": name of dataset
        - "n_samples": number of samples
        - "n_features": number of features (2)
        - "n_classes": number of classes (2)
        - "class_distribution": count of each class
    """
    
    np.random.seed(random_seed)
    
    n_minority = int(n_samples * imbalance_ratio)
    n_majority = n_samples - n_minority
    
    # Majority class (class 0)
    X_majority = np.random.randn(n_majority, 2) - 1.5
    y_majority = np.zeros(n_majority, dtype=int)
    
    # Minority class (class 1)
    X_minority = np.random.randn(n_minority, 2) + 1.5
    y_minority = np.ones(n_minority, dtype=int)
    
    # Combine and shuffle
    X = np.vstack([X_majority, X_minority])
    y = np.hstack([y_majority, y_minority])
    
    shuffle_idx = np.random.permutation(len(X))
    X = X[shuffle_idx]
    y = y[shuffle_idx]
    
    # Count class distribution
    unique, counts = np.unique(y, return_counts=True)
    class_dist = {int(u): int(c) for u, c in zip(unique, counts)}
    
    return {
        "X": X,
        "y": y,
        "dataset_type": "Imbalanced Binary Classification",
        "n_samples": len(X),
        "n_features": 2,
        "n_classes": 2,
        "class_distribution": class_dist,
        "imbalance_ratio": imbalance_ratio
    }


def generate_high_dimensional_dataset(n_samples=100, n_features=50, n_informative=5, random_seed=42):
    """
    Generate a high-dimensional dataset.
    
    Creates a dataset with many features where only a subset are informative
    for the classification task. Useful for testing feature selection.
    
    Parameters:
    -----------
    n_samples : int, optional
        Number of samples to generate (default: 100)
    n_features : int, optional
        Total number of features (default: 50)
    n_informative : int, optional
        Number of informative features (default: 5)
    random_seed : int, optional
        Random seed for reproducibility (default: 42)
    
    Returns:
    --------
    dict
        Dictionary containing:
        - "X": feature matrix (n_samples x n_features)
        - "y": binary labels (0 or 1)
        - "dataset_type": name of dataset
        - "n_samples": number of samples
        - "n_features": number of features
        - "n_informative": number of informative features
        - "n_classes": number of classes (2)
        - "informative_indices": indices of informative features
    """
    
    np.random.seed(random_seed)
    
    # Generate informative features
    X_informative = np.random.randn(n_samples, n_informative)
    
    # Generate labels based on informative features
    y = (np.sum(X_informative, axis=1) > 0).astype(int)
    
    # Generate noise features
    n_noise = n_features - n_informative
    X_noise = np.random.randn(n_samples, n_noise)
    
    # Combine features
    X = np.hstack([X_informative, X_noise])
    
    # Shuffle feature order
    feature_idx = np.random.permutation(n_features)
    X = X[:, feature_idx]
    informative_indices = np.where(np.isin(feature_idx, np.arange(n_informative)))[0].tolist()
    
    return {
        "X": X,
        "y": y,
        "dataset_type": "High-Dimensional",
        "n_samples": n_samples,
        "n_features": n_features,
        "n_informative": n_informative,
        "n_classes": 2,
        "informative_indices": informative_indices
    }


def generate_clustered_dataset(n_samples=300, n_clusters=4, n_features=2, random_seed=42):
    """
    Generate a dataset with distinct clusters for clustering tasks.
    
    Parameters:
    -----------
    n_samples : int, optional
        Number of samples to generate (default: 300)
    n_clusters : int, optional
        Number of clusters (default: 4)
    n_features : int, optional
        Number of features per sample (default: 2)
    random_seed : int, optional
        Random seed for reproducibility (default: 42)
    
    Returns:
    --------
    dict
        Dictionary containing:
        - "X": feature matrix (n_samples x n_features)
        - "cluster_labels": true cluster assignment
        - "cluster_centers": center of each cluster
        - "dataset_type": name of dataset
        - "n_samples": number of samples
        - "n_features": number of features
        - "n_clusters": number of clusters
    """
    
    np.random.seed(random_seed)
    
    n_samples_per_cluster = n_samples // n_clusters
    X_list = []
    labels_list = []
    centers_list = []
    
    # Generate clusters
    for cluster_idx in range(n_clusters):
        # Random cluster center
        center = np.random.uniform(-10, 10, n_features)
        centers_list.append(center)
        
        # Generate samples around center
        X_cluster = np.random.randn(n_samples_per_cluster, n_features) * 0.5 + center
        labels_cluster = np.full(n_samples_per_cluster, cluster_idx, dtype=int)
        
        X_list.append(X_cluster)
        labels_list.append(labels_cluster)
    
    # Combine and shuffle
    X = np.vstack(X_list)
    cluster_labels = np.hstack(labels_list)
    
    shuffle_idx = np.random.permutation(len(X))
    X = X[shuffle_idx]
    cluster_labels = cluster_labels[shuffle_idx]
    
    return {
        "X": X,
        "cluster_labels": cluster_labels,
        "cluster_centers": np.array(centers_list),
        "dataset_type": "Clustered",
        "n_samples": len(X),
        "n_features": n_features,
        "n_clusters": n_clusters
    }


def generate_spiral_dataset(n_samples=300, n_arms=2, noise=0.1, random_seed=42):
    """
    Generate a spiral dataset (highly non-linear).
    
    Creates a dataset with samples arranged in spiral patterns, useful for
    testing non-linear classifiers.
    
    Parameters:
    -----------
    n_samples : int, optional
        Number of samples to generate (default: 300)
    n_arms : int, optional
        Number of spiral arms (default: 2)
    noise : float, optional
        Standard deviation of noise (default: 0.1)
    random_seed : int, optional
        Random seed for reproducibility (default: 42)
    
    Returns:
    --------
    dict
        Dictionary containing:
        - "X": feature matrix (n_samples x 2)
        - "y": class labels (0 to n_arms-1)
        - "dataset_type": name of dataset
        - "n_samples": number of samples
        - "n_features": number of features (2)
        - "n_classes": number of classes
    """
    
    np.random.seed(random_seed)
    
    n_samples_per_arm = n_samples // n_arms
    X_list = []
    y_list = []
    
    for arm_idx in range(n_arms):
        # Parameter t goes from 0 to 4*pi
        t = np.linspace(0, 4 * pi, n_samples_per_arm)
        
        # Offset arm by 2*pi/n_arms
        t = t + (2 * pi * arm_idx) / n_arms
        
        # Generate spiral coordinates
        r = t
        x = r * np.cos(t)
        y = r * np.sin(t)
        
        # Stack and add noise
        X_arm = np.column_stack([x, y])
        X_arm += np.random.randn(n_samples_per_arm, 2) * noise
        
        X_list.append(X_arm)
        y_list.append(np.full(n_samples_per_arm, arm_idx, dtype=int))
    
    # Combine and shuffle
    X = np.vstack(X_list)
    y = np.hstack(y_list)
    
    shuffle_idx = np.random.permutation(len(X))
    X = X[shuffle_idx]
    y = y[shuffle_idx]
    
    return {
        "X": X,
        "y": y,
        "dataset_type": "Spiral (Non-linear)",
        "n_samples": len(X),
        "n_features": 2,
        "n_classes": n_arms
    }


def print_dataset_summary(dataset):
    """
    Print a summary of a dataset.
    
    Parameters:
    -----------
    dataset : dict
        Dataset dictionary from generation functions
    """
    
    print("\n" + "=" * 70)
    print("DATASET SUMMARY: {}".format(dataset.get("dataset_type", "Unknown")))
    print("=" * 70)
    print("\nDataset Information:")
    print("  - Number of samples: {}".format(dataset["n_samples"]))
    print("  - Number of features: {}".format(dataset["n_features"]))
    
    if "n_classes" in dataset:
        print("  - Number of classes: {}".format(dataset["n_classes"]))
    
    if "n_clusters" in dataset:
        print("  - Number of clusters: {}".format(dataset["n_clusters"]))
    
    if "n_informative" in dataset:
        print("  - Informative features: {}".format(dataset["n_informative"]))
    
    if "class_distribution" in dataset:
        print("  - Class distribution: {}".format(dataset["class_distribution"]))
    
    if "imbalance_ratio" in dataset:
        print("  - Imbalance ratio: {:.2%}".format(dataset["imbalance_ratio"]))
    
    print("\nFeature Statistics:")
    X = dataset["X"]
    print("  - Feature mean: [{:.4f}, {:.4f}, ...]".format(X[:, 0].mean(), X[:, 1].mean() if X.shape[1] > 1 else 0))
    print("  - Feature std:  [{:.4f}, {:.4f}, ...]".format(X[:, 0].std(), X[:, 1].std() if X.shape[1] > 1 else 0))
    print("  - Feature min:  [{:.4f}, {:.4f}, ...]".format(X[:, 0].min(), X[:, 1].min() if X.shape[1] > 1 else 0))
    print("  - Feature max:  [{:.4f}, {:.4f}, ...]".format(X[:, 0].max(), X[:, 1].max() if X.shape[1] > 1 else 0))
    
    if "y" in dataset:
        y = dataset["y"]
        print("\nTarget Statistics:")
        print("  - Target type: {}".format("Classification" if len(np.unique(y)) < 20 else "Regression"))
        print("  - Unique values: {}".format(len(np.unique(y))))


def main():
    """
    Main function demonstrating synthetic data generation.
    """
    print("\n")
    print("#" * 70)
    print("#" + " " * 68 + "#")
    print("#" + "  SYNTHETIC DATASET GENERATION DEMONSTRATION".center(68) + "#")
    print("#" + " " * 68 + "#")
    print("#" * 70)
    
    # 1. Linearly Separable Dataset
    print("\n\nEXAMPLE 1: LINEARLY SEPARABLE DATASET")
    data1 = generate_linearly_separable_data(n_samples=200, n_features=2)
    print_dataset_summary(data1)
    
    # 2. XOR Dataset
    print("\n\nEXAMPLE 2: XOR (NON-LINEAR) DATASET")
    data2 = generate_xor_dataset(n_samples=200, noise=0.05)
    print_dataset_summary(data2)
    
    # 3. Multi-class Dataset
    print("\n\nEXAMPLE 3: MULTI-CLASS DATASET")
    data3 = generate_multiclass_dataset(n_samples=300, n_features=3, n_classes=4)
    print_dataset_summary(data3)
    
    # 4. Regression Dataset
    print("\n\nEXAMPLE 4: REGRESSION DATASET")
    data4 = generate_regression_dataset(n_samples=200, n_features=2, noise_level=0.2)
    print_dataset_summary(data4)
    
    # 5. Imbalanced Dataset
    print("\n\nEXAMPLE 5: IMBALANCED BINARY CLASSIFICATION")
    data5 = generate_imbalanced_dataset(n_samples=200, imbalance_ratio=0.15)
    print_dataset_summary(data5)
    
    # 6. High-Dimensional Dataset
    print("\n\nEXAMPLE 6: HIGH-DIMENSIONAL DATASET")
    data6 = generate_high_dimensional_dataset(n_samples=100, n_features=100, n_informative=5)
    print_dataset_summary(data6)
    print("  - Informative feature indices: {}".format(data6["informative_indices"][:5]))
    
    # 7. Clustered Dataset
    print("\n\nEXAMPLE 7: CLUSTERED DATASET")
    data7 = generate_clustered_dataset(n_samples=300, n_clusters=5, n_features=2)
    print_dataset_summary(data7)
    
    # 8. Spiral Dataset
    print("\n\nEXAMPLE 8: SPIRAL DATASET (HIGHLY NON-LINEAR)")
    data8 = generate_spiral_dataset(n_samples=300, n_arms=3, noise=0.05)
    print_dataset_summary(data8)
    
    # Summary
    print("\n\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("\nSynthetic Dataset Generation Functions Available:")
    print("  1. generate_linearly_separable_data() - Linearly separable classification")
    print("  2. generate_xor_dataset() - Non-linear XOR classification")
    print("  3. generate_multiclass_dataset() - Multi-class classification")
    print("  4. generate_regression_dataset() - Regression with noise")
    print("  5. generate_imbalanced_dataset() - Class imbalance simulation")
    print("  6. generate_high_dimensional_dataset() - Feature selection testing")
    print("  7. generate_clustered_dataset() - Unsupervised clustering")
    print("  8. generate_spiral_dataset() - Complex non-linear patterns")
    print("\nTotal datasets generated: 8")
    print("=" * 70)


if __name__ == "__main__":
    main()
