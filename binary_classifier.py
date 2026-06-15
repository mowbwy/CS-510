"""
Binary Classifier Implementation using Neuron
==============================================
This module implements a binary classifier that uses the neuron function
to make classification predictions. The classifier supports configurable
activation functions, custom weights, and dataset handling.

Author: Binary Classifier Module
Version: 1.0
"""

import numpy as np
from math import sqrt, exp


def sigmoid(z):
    """
    Sigmoid activation function.
    
    Computes the sigmoid activation: 1 / (1 + e^(-z))
    Output range: (0, 1)
    
    Parameters:
    -----------
    z : float or array-like
        Input value(s) to apply sigmoid activation
    
    Returns:
    --------
    float or array
        Sigmoid output value(s)
    """
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))


def relu(z):
    """
    Rectified Linear Unit (ReLU) activation function.
    
    Computes ReLU activation: max(0, z)
    Output range: [0, infinity)
    
    Parameters:
    -----------
    z : float or array-like
        Input value(s) to apply ReLU activation
    
    Returns:
    --------
    float or array
        ReLU output value(s)
    """
    return np.maximum(0, z)


def neuron(inputs, weights, bias, activation='sigmoid'):
    """
    Simulates a single neuron with adjustable activation function.
    
    Performs forward propagation: output = activation(inputs · weights + bias)
    
    Parameters:
    -----------
    inputs : array-like
        Input values to the neuron (shape: n_features,)
    weights : array-like
        Weight values for each input (shape: n_features,)
    bias : float
        Bias term for the neuron
    activation : str, optional
        Activation function to use. Options: 'sigmoid' or 'relu'
        Default is 'sigmoid'
    
    Returns:
    --------
    float
        Output of the neuron after applying the activation function
    
    Raises:
    -------
    ValueError
        If activation function is not recognized or shapes don't match
    """
    
    # Convert inputs and weights to numpy arrays
    inputs = np.asarray(inputs)
    weights = np.asarray(weights)
    
    # Validate shapes match
    if inputs.shape != weights.shape:
        raise ValueError(
            "Inputs shape {} must match weights shape {}".format(
                inputs.shape, weights.shape
            )
        )
    
    # Calculate weighted sum (z = w·x + b)
    z = np.dot(weights, inputs) + bias
    
    # Apply activation function
    if activation == 'sigmoid':
        return sigmoid(z)
    elif activation == 'relu':
        return relu(z)
    else:
        raise ValueError(
            "Unknown activation function: '{}'. Supported options are: 'sigmoid', 'relu'".format(
                activation
            )
        )


def binary_classifier(dataset, weights, bias=0.0, activation='sigmoid', threshold=0.5):
    """
    Binary classifier using a single neuron.
    
    Classifies each sample in the dataset as class 0 or class 1 based on
    the neuron output and a classification threshold.
    
    Parameters:
    -----------
    dataset : array-like (N x M)
        Input dataset where N is number of samples and M is number of features
    weights : array-like (M,)
        Weight vector for each feature. Must match number of features in dataset.
    bias : float, optional
        Bias term for the neuron. Default is 0.0
    activation : str, optional
        Activation function to use: 'sigmoid' or 'relu'. Default is 'sigmoid'
    threshold : float, optional
        Classification threshold. Default is 0.5
        - If neuron_output >= threshold: predict class 1
        - If neuron_output < threshold: predict class 0
    
    Returns:
    --------
    dict
        Dictionary containing:
        - "predictions": array of predictions (0 or 1)
        - "probabilities": array of raw neuron outputs
        - "accuracy": accuracy if true_labels provided
        - "metrics": classification metrics (precision, recall, F1)
    
    Raises:
    -------
    ValueError
        If weights length doesn't match dataset features
    TypeError
        If dataset or weights have invalid types
    """
    
    # Convert dataset to numpy array
    dataset = np.asarray(dataset, dtype=float)
    weights = np.asarray(weights, dtype=float)
    
    # Validate inputs
    if dataset.ndim == 1:
        dataset = dataset.reshape(1, -1)
    
    if dataset.ndim != 2:
        raise ValueError("Dataset must be 2D array (samples x features)")
    
    n_samples, n_features = dataset.shape
    
    if len(weights) != n_features:
        raise ValueError(
            "Number of weights ({}) must match number of features ({})".format(
                len(weights), n_features
            )
        )
    
    print("=" * 70)
    print("BINARY CLASSIFIER USING NEURON")
    print("=" * 70)
    print("\nDataset Configuration:")
    print("  - Number of samples: {}".format(n_samples))
    print("  - Number of features: {}".format(n_features))
    print("  - Activation function: {}".format(activation))
    print("  - Bias: {}".format(bias))
    print("  - Classification threshold: {}".format(threshold))
    print("  - Weights: {}".format(weights))
    
    # Make predictions for each sample
    probabilities = []
    predictions = []
    
    for i in range(n_samples):
        sample = dataset[i, :]
        # Get neuron output (probability)
        prob = neuron(sample, weights, bias, activation=activation)
        probabilities.append(prob)
        
        # Apply threshold for classification
        prediction = 1 if prob >= threshold else 0
        predictions.append(prediction)
    
    probabilities = np.array(probabilities)
    predictions = np.array(predictions)
    
    result = {
        "predictions": predictions,
        "probabilities": probabilities,
        "n_samples": n_samples,
        "n_features": n_features,
        "activation": activation,
        "threshold": threshold
    }
    
    return result


def evaluate_classifier(predictions, true_labels):
    """
    Evaluate binary classifier performance.
    
    Parameters:
    -----------
    predictions : array-like
        Predicted class labels (0 or 1)
    true_labels : array-like
        True class labels (0 or 1)
    
    Returns:
    --------
    dict
        Dictionary containing evaluation metrics:
        - "accuracy": (TP + TN) / (TP + TN + FP + FN)
        - "precision": TP / (TP + FP)
        - "recall": TP / (TP + FN)
        - "f1_score": 2 * (precision * recall) / (precision + recall)
        - "true_positives": TP count
        - "true_negatives": TN count
        - "false_positives": FP count
        - "false_negatives": FN count
    """
    
    predictions = np.asarray(predictions)
    true_labels = np.asarray(true_labels)
    
    if len(predictions) != len(true_labels):
        raise ValueError("Predictions and true labels must have same length")
    
    # Calculate confusion matrix
    tp = np.sum((predictions == 1) & (true_labels == 1))
    tn = np.sum((predictions == 0) & (true_labels == 0))
    fp = np.sum((predictions == 1) & (true_labels == 0))
    fn = np.sum((predictions == 0) & (true_labels == 1))
    
    # Calculate metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print("\n" + "=" * 70)
    print("CLASSIFICATION METRICS")
    print("=" * 70)
    print("\nConfusion Matrix:")
    print("  True Positives (TP): {}".format(tp))
    print("  True Negatives (TN): {}".format(tn))
    print("  False Positives (FP): {}".format(fp))
    print("  False Negatives (FN): {}".format(fn))
    print("\nPerformance Metrics:")
    print("  Accuracy:  {:.4f}".format(accuracy))
    print("  Precision: {:.4f}".format(precision))
    print("  Recall:    {:.4f}".format(recall))
    print("  F1 Score:  {:.4f}".format(f1))
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "true_positives": tp,
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn
    }


def cross_validate_classifier(dataset, true_labels, weights, bias=0.0, 
                             activation='sigmoid', threshold=0.5, k_folds=5):
    """
    Perform k-fold cross-validation on the classifier.
    
    Parameters:
    -----------
    dataset : array-like (N x M)
        Input dataset
    true_labels : array-like (N,)
        True class labels
    weights : array-like (M,)
        Weight vector
    bias : float, optional
        Bias term
    activation : str, optional
        Activation function
    threshold : float, optional
        Classification threshold
    k_folds : int, optional
        Number of folds for cross-validation
    
    Returns:
    --------
    dict
        Dictionary containing cross-validation results
    """
    
    dataset = np.asarray(dataset, dtype=float)
    true_labels = np.asarray(true_labels)
    
    if len(dataset) != len(true_labels):
        raise ValueError("Dataset and labels must have same number of samples")
    
    print("\n" + "=" * 70)
    print("K-FOLD CROSS-VALIDATION (k={})".format(k_folds))
    print("=" * 70)
    
    n_samples = len(dataset)
    fold_size = n_samples // k_folds
    
    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []
    
    for fold in range(k_folds):
        # Split into train and test
        test_start = fold * fold_size
        test_end = test_start + fold_size if fold < k_folds - 1 else n_samples
        
        test_indices = np.arange(test_start, test_end)
        train_indices = np.concatenate([
            np.arange(0, test_start),
            np.arange(test_end, n_samples)
        ])
        
        X_test = dataset[test_indices]
        y_test = true_labels[test_indices]
        
        # Make predictions
        result = binary_classifier(
            X_test, weights, bias, activation, threshold
        )
        predictions = result["predictions"]
        
        # Evaluate
        metrics = evaluate_classifier(predictions, y_test)
        
        accuracies.append(metrics["accuracy"])
        precisions.append(metrics["precision"])
        recalls.append(metrics["recall"])
        f1_scores.append(metrics["f1_score"])
        
        print("\nFold {}: Accuracy = {:.4f}".format(fold + 1, metrics["accuracy"]))
    
    print("\n" + "=" * 70)
    print("CROSS-VALIDATION SUMMARY")
    print("=" * 70)
    print("\nMean Accuracy:  {:.4f} (+/- {:.4f})".format(
        np.mean(accuracies), np.std(accuracies)
    ))
    print("Mean Precision: {:.4f} (+/- {:.4f})".format(
        np.mean(precisions), np.std(precisions)
    ))
    print("Mean Recall:    {:.4f} (+/- {:.4f})".format(
        np.mean(recalls), np.std(recalls)
    ))
    print("Mean F1 Score:  {:.4f} (+/- {:.4f})".format(
        np.mean(f1_scores), np.std(f1_scores)
    ))
    
    return {
        "accuracies": accuracies,
        "precisions": precisions,
        "recalls": recalls,
        "f1_scores": f1_scores,
        "mean_accuracy": np.mean(accuracies),
        "std_accuracy": np.std(accuracies),
        "mean_f1": np.mean(f1_scores),
        "std_f1": np.std(f1_scores)
    }


def main():
    """
    Main function demonstrating binary classifier functionality.
    """
    print("\n")
    print("#" * 70)
    print("#" + " " * 68 + "#")
    print("#" + "  BINARY CLASSIFIER USING NEURON IMPLEMENTATION".center(68) + "#")
    print("#" + " " * 68 + "#")
    print("#" * 70)
    
    # Example 1: Simple linearly separable dataset
    print("\n\n" + "=" * 70)
    print("EXAMPLE 1: SIMPLE DATASET WITH SIGMOID ACTIVATION")
    print("=" * 70)
    
    # Create a simple dataset
    X_simple = np.array([
        [1.0, 2.0],
        [2.0, 3.0],
        [3.0, 1.0],
        [5.0, 1.0],
        [6.0, 0.0],
        [7.0, 1.0]
    ])
    
    y_simple = np.array([0, 0, 0, 1, 1, 1])
    
    # Define weights (manually chosen for this example)
    weights_simple = np.array([0.5, 0.3])
    bias_simple = -1.0
    
    # Make predictions
    result_simple = binary_classifier(
        X_simple, 
        weights_simple, 
        bias_simple, 
        activation='sigmoid',
        threshold=0.5
    )
    
    print("\nSample-wise Predictions:")
    print("-" * 70)
    for i in range(len(X_simple)):
        print("Sample {}: Features = {}, Probability = {:.4f}, Prediction = {}".format(
            i + 1, list(X_simple[i]), result_simple["probabilities"][i], 
            result_simple["predictions"][i]
        ))
    
    # Evaluate
    metrics_simple = evaluate_classifier(result_simple["predictions"], y_simple)
    
    # Example 2: Using ReLU activation
    print("\n\n" + "=" * 70)
    print("EXAMPLE 2: SAME DATASET WITH RELU ACTIVATION")
    print("=" * 70)
    
    result_relu = binary_classifier(
        X_simple,
        weights_simple,
        bias_simple,
        activation='relu',
        threshold=0.5
    )
    
    print("\nSample-wise Predictions:")
    print("-" * 70)
    for i in range(len(X_simple)):
        print("Sample {}: Features = {}, Output = {:.4f}, Prediction = {}".format(
            i + 1, list(X_simple[i]), result_relu["probabilities"][i],
            result_relu["predictions"][i]
        ))
    
    metrics_relu = evaluate_classifier(result_relu["predictions"], y_simple)
    
    # Example 3: Larger dataset with cross-validation
    print("\n\n" + "=" * 70)
    print("EXAMPLE 3: LARGER DATASET WITH CROSS-VALIDATION")
    print("=" * 70)
    
    # Create a larger synthetic dataset
    np.random.seed(42)
    n_samples = 100
    n_features = 3
    
    X_large = np.random.randn(n_samples, n_features)
    # Create labels based on a linear combination of features
    y_large = (X_large[:, 0] + 0.5 * X_large[:, 1] - 0.3 * X_large[:, 2] > 0).astype(int)
    
    weights_large = np.array([0.4, 0.3, -0.2])
    bias_large = 0.0
    
    # Perform cross-validation
    cv_results = cross_validate_classifier(
        X_large,
        y_large,
        weights_large,
        bias_large,
        activation='sigmoid',
        threshold=0.5,
        k_folds=5
    )
    
    # Example 4: Different threshold comparison
    print("\n\n" + "=" * 70)
    print("EXAMPLE 4: EFFECT OF DIFFERENT CLASSIFICATION THRESHOLDS")
    print("=" * 70)
    
    thresholds = [0.3, 0.5, 0.7]
    
    for thresh in thresholds:
        result = binary_classifier(
            X_simple,
            weights_simple,
            bias_simple,
            activation='sigmoid',
            threshold=thresh
        )
        metrics = evaluate_classifier(result["predictions"], y_simple)
        print("\nThreshold = {}: Accuracy = {:.4f}, F1 = {:.4f}".format(
            thresh, metrics["accuracy"], metrics["f1_score"]
        ))
    
    # Summary
    print("\n\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("\nBinary Classifier Features Demonstrated:")
    print("  1. Neuron-based classification with multiple samples")
    print("  2. Multiple activation functions (sigmoid, relu)")
    print("  3. Customizable weights parameter")
    print("  4. Dataset input in array format")
    print("  5. Configurable classification threshold")
    print("  6. Performance metrics calculation")
    print("  7. K-fold cross-validation")
    print("  8. Threshold sensitivity analysis")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
