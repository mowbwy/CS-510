"""
Weight Calculation and Optimization Module
===========================================
This module provides various algorithms to calculate and optimize weights
for neural network classifiers based on datasets and selected activation functions.

Author: Weight Calculation Module
Version: 1.0
"""

import numpy as np
from scipy import optimize
from math import sqrt, exp


def sigmoid(z):
    """
    Sigmoid activation function.
    
    Parameters:
    -----------
    z : float or array-like
        Input value(s)
    
    Returns:
    --------
    float or array
        Sigmoid output
    """
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))


def sigmoid_derivative(z):
    """
    Derivative of sigmoid activation function.
    
    Parameters:
    -----------
    z : float or array-like
        Input value(s)
    
    Returns:
    --------
    float or array
        Sigmoid derivative output
    """
    s = sigmoid(z)
    return s * (1 - s)


def relu(z):
    """
    ReLU activation function.
    
    Parameters:
    -----------
    z : float or array-like
        Input value(s)
    
    Returns:
    --------
    float or array
        ReLU output
    """
    return np.maximum(0, z)


def relu_derivative(z):
    """
    Derivative of ReLU activation function.
    
    Parameters:
    -----------
    z : float or array-like
        Input value(s)
    
    Returns:
    --------
    float or array
        ReLU derivative output
    """
    return (z > 0).astype(float)


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


def calculate_weights_random(dataset, activation='sigmoid', random_seed=42):
    """
    Calculate weights using random initialization with variance scaling.
    
    This is a baseline approach that initializes weights randomly with
    proper variance scaling based on the activation function.
    
    Parameters:
    -----------
    dataset : dict
        Dataset dictionary with keys:
        - "X": feature matrix (n_samples x n_features)
        - "y": labels (n_samples,)
    activation : str, optional
        Activation function: 'sigmoid' or 'relu' (default: 'sigmoid')
    random_seed : int, optional
        Random seed for reproducibility (default: 42)
    
    Returns:
    --------
    dict
        Dictionary containing:
        - "weights": calculated weight vector
        - "bias": bias term
        - "method": weight calculation method
        - "activation": activation function used
        - "n_features": number of features
    
    Raises:
    -------
    ValueError
        If activation function is not recognized
    """
    
    if activation not in ['sigmoid', 'relu']:
        raise ValueError("Activation must be 'sigmoid' or 'relu'")
    
    np.random.seed(random_seed)
    
    X = dataset["X"]
    n_samples, n_features = X.shape
    
    print("=" * 70)
    print("WEIGHT CALCULATION: Random Initialization")
    print("=" * 70)
    print("\nDataset Information:")
    print("  - Number of samples: {}".format(n_samples))
    print("  - Number of features: {}".format(n_features))
    print("  - Activation function: {}".format(activation))
    
    # Variance scaling based on activation function
    if activation == 'sigmoid':
        variance = 1.0 / n_features
    else:  # relu
        variance = 2.0 / n_features
    
    # Initialize weights
    weights = np.random.randn(n_features) * sqrt(variance)
    bias = 0.0
    
    print("\nWeight Initialization:")
    print("  - Variance scaling: {:.4f}".format(variance))
    print("  - Weights: {}".format(weights))
    print("  - Bias: {}".format(bias))
    
    return {
        "weights": weights,
        "bias": bias,
        "method": "Random Initialization",
        "activation": activation,
        "n_features": n_features
    }


def calculate_weights_hebbian(dataset, activation='sigmoid', learning_rate=0.01, epochs=100):
    """
    Calculate weights using Hebbian learning rule.
    
    Implements the basic Hebbian learning principle: neurons that fire together
    wire together. Updates weights based on correlation between inputs and outputs.
    
    Parameters:
    -----------
    dataset : dict
        Dataset dictionary with keys:
        - "X": feature matrix (n_samples x n_features)
        - "y": labels (n_samples,)
    activation : str, optional
        Activation function: 'sigmoid' or 'relu' (default: 'sigmoid')
    learning_rate : float, optional
        Learning rate for weight updates (default: 0.01)
    epochs : int, optional
        Number of training epochs (default: 100)
    
    Returns:
    --------
    dict
        Dictionary containing:
        - "weights": learned weight vector
        - "bias": learned bias term
        - "method": weight calculation method
        - "activation": activation function used
        - "loss_history": MSE loss at each epoch
        - "learning_rate": learning rate used
    
    Raises:
    -------
    ValueError
        If activation function is not recognized
    """
    
    if activation not in ['sigmoid', 'relu']:
        raise ValueError("Activation must be 'sigmoid' or 'relu'")
    
    X = dataset["X"]
    y = dataset["y"].reshape(-1, 1)
    n_samples, n_features = X.shape
    
    print("=" * 70)
    print("WEIGHT CALCULATION: Hebbian Learning Rule")
    print("=" * 70)
    print("\nDataset Information:")
    print("  - Number of samples: {}".format(n_samples))
    print("  - Number of features: {}".format(n_features))
    print("  - Activation function: {}".format(activation))
    print("  - Learning rate: {}".format(learning_rate))
    print("  - Epochs: {}".format(epochs))
    
    # Initialize weights and bias
    weights = np.zeros(n_features)
    bias = 0.0
    loss_history = []
    
    # Select activation function
    if activation == 'sigmoid':
        activation_fn = sigmoid
        activation_deriv = sigmoid_derivative
    else:
        activation_fn = relu
        activation_deriv = relu_derivative
    
    print("\nTraining Progress:")
    
    # Training loop
    for epoch in range(epochs):
        # Forward pass
        z = X.dot(weights) + bias
        output = activation_fn(z)
        
        # Calculate loss
        loss = np.mean((output - y) ** 2)
        loss_history.append(loss)
        
        # Hebbian learning: weight update based on pre and post synaptic activity
        for i in range(n_samples):
            error = y[i] - output[i]
            weights += learning_rate * error * X[i]
            bias += learning_rate * error
        
        if (epoch + 1) % (epochs // 10) == 0 or epoch == 0:
            print("  Epoch {:3d}/{}: Loss = {:.6f}".format(epoch + 1, epochs, loss))
    
    print("\nFinal Training Metrics:")
    print("  - Final Loss: {:.6f}".format(loss_history[-1]))
    print("  - Loss reduction: {:.6f}".format(loss_history[0] - loss_history[-1]))
    
    return {
        "weights": weights,
        "bias": bias,
        "method": "Hebbian Learning",
        "activation": activation,
        "loss_history": loss_history,
        "learning_rate": learning_rate,
        "n_features": n_features
    }


def calculate_weights_gradient_descent(dataset, activation='sigmoid', learning_rate=0.01, epochs=100):
    """
    Calculate weights using Gradient Descent optimization.
    
    Implements standard gradient descent to minimize MSE loss by updating
    weights in the direction of negative gradient.
    
    Parameters:
    -----------
    dataset : dict
        Dataset dictionary with keys:
        - "X": feature matrix (n_samples x n_features)
        - "y": labels (n_samples,)
    activation : str, optional
        Activation function: 'sigmoid' or 'relu' (default: 'sigmoid')
    learning_rate : float, optional
        Learning rate for weight updates (default: 0.01)
    epochs : int, optional
        Number of training epochs (default: 100)
    
    Returns:
    --------
    dict
        Dictionary containing:
        - "weights": learned weight vector
        - "bias": learned bias term
        - "method": weight calculation method
        - "activation": activation function used
        - "loss_history": MSE loss at each epoch
        - "learning_rate": learning rate used
    
    Raises:
    -------
    ValueError
        If activation function is not recognized
    """
    
    if activation not in ['sigmoid', 'relu']:
        raise ValueError("Activation must be 'sigmoid' or 'relu'")
    
    X = dataset["X"]
    y = dataset["y"].reshape(-1, 1)
    n_samples, n_features = X.shape
    
    print("=" * 70)
    print("WEIGHT CALCULATION: Gradient Descent")
    print("=" * 70)
    print("\nDataset Information:")
    print("  - Number of samples: {}".format(n_samples))
    print("  - Number of features: {}".format(n_features))
    print("  - Activation function: {}".format(activation))
    print("  - Learning rate: {}".format(learning_rate))
    print("  - Epochs: {}".format(epochs))
    
    # Initialize weights and bias
    weights = np.zeros(n_features)
    bias = 0.0
    loss_history = []
    
    # Select activation function
    if activation == 'sigmoid':
        activation_fn = sigmoid
        activation_deriv = sigmoid_derivative
    else:
        activation_fn = relu
        activation_deriv = relu_derivative
    
    print("\nTraining Progress:")
    
    # Training loop
    for epoch in range(epochs):
        # Forward pass
        z = X.dot(weights) + bias
        output = activation_fn(z)
        
        # Calculate loss
        loss = np.mean((output - y) ** 2)
        loss_history.append(loss)
        
        # Backward pass - compute gradients
        error = output - y
        d_output = error * activation_deriv(z)
        
        # Update weights and bias
        dw = (2 / n_samples) * X.T.dot(d_output)
        db = (2 / n_samples) * np.sum(d_output)
        
        weights -= learning_rate * dw
        bias -= learning_rate * db
        
        if (epoch + 1) % (epochs // 10) == 0 or epoch == 0:
            print("  Epoch {:3d}/{}: Loss = {:.6f}".format(epoch + 1, epochs, loss))
    
    print("\nFinal Training Metrics:")
    print("  - Final Loss: {:.6f}".format(loss_history[-1]))
    print("  - Loss reduction: {:.6f}".format(loss_history[0] - loss_history[-1]))
    
    return {
        "weights": weights,
        "bias": bias,
        "method": "Gradient Descent",
        "activation": activation,
        "loss_history": loss_history,
        "learning_rate": learning_rate,
        "n_features": n_features
    }


def calculate_weights_numerical_gradient(dataset, activation='sigmoid', max_iterations=1000):
    """
    Calculate weights using Numerical Gradient and L-BFGS-B optimizer.
    
    Uses finite differences to compute gradients and scipy's L-BFGS-B optimizer
    for weight optimization. More sophisticated than basic gradient descent.
    
    Parameters:
    -----------
    dataset : dict
        Dataset dictionary with keys:
        - "X": feature matrix (n_samples x n_features)
        - "y": labels (n_samples,)
    activation : str, optional
        Activation function: 'sigmoid' or 'relu' (default: 'sigmoid')
    max_iterations : int, optional
        Maximum number of optimizer iterations (default: 1000)
    
    Returns:
    --------
    dict
        Dictionary containing:
        - "weights": optimized weight vector
        - "bias": optimized bias term
        - "method": weight calculation method
        - "activation": activation function used
        - "final_loss": final MSE loss value
        - "iterations": number of iterations used
        - "success": whether optimization succeeded
    
    Raises:
    -------
    ValueError
        If activation function is not recognized
    """
    
    if activation not in ['sigmoid', 'relu']:
        raise ValueError("Activation must be 'sigmoid' or 'relu'")
    
    X = dataset["X"]
    y = dataset["y"].reshape(-1, 1)
    n_samples, n_features = X.shape
    
    print("=" * 70)
    print("WEIGHT CALCULATION: Numerical Gradient + L-BFGS-B Optimizer")
    print("=" * 70)
    print("\nDataset Information:")
    print("  - Number of samples: {}".format(n_samples))
    print("  - Number of features: {}".format(n_features))
    print("  - Activation function: {}".format(activation))
    print("  - Max iterations: {}".format(max_iterations))
    
    # Select activation function
    if activation == 'sigmoid':
        activation_fn = sigmoid
    else:
        activation_fn = relu
    
    # Combine weights and bias into single vector
    initial_params = np.zeros(n_features + 1)
    
    def loss_function(params):
        """Calculate MSE loss for given parameters"""
        w = params[:n_features]
        b = params[n_features]
        z = X.dot(w) + b
        output = activation_fn(z)
        loss = np.mean((output - y) ** 2)
        return loss
    
    def gradient_function(params):
        """Calculate numerical gradients using finite differences"""
        eps = 1e-7
        grad = np.zeros_like(params)
        for i in range(len(params)):
            params_plus = params.copy()
            params_minus = params.copy()
            params_plus[i] += eps
            params_minus[i] -= eps
            grad[i] = (loss_function(params_plus) - loss_function(params_minus)) / (2 * eps)
        return grad
    
    print("\nOptimization Progress:")
    
    # Optimize using L-BFGS-B
    result = optimize.minimize(
        loss_function,
        initial_params,
        method='L-BFGS-B',
        jac=gradient_function,
        options={'maxiter': max_iterations, 'disp': False}
    )
    
    # Extract weights and bias
    weights = result.x[:n_features]
    bias = result.x[n_features]
    final_loss = loss_function(result.x)
    
    print("Optimization Results:")
    print("  - Success: {}".format(result.success))
    print("  - Final Loss: {:.6f}".format(final_loss))
    print("  - Iterations: {}".format(result.nit))
    
    return {
        "weights": weights,
        "bias": bias,
        "method": "Numerical Gradient + L-BFGS-B",
        "activation": activation,
        "final_loss": final_loss,
        "iterations": result.nit,
        "success": result.success,
        "n_features": n_features
    }


def calculate_weights_pearson_correlation(dataset, activation='sigmoid'):
    """
    Calculate weights using Pearson correlation coefficients.
    
    Simple statistical approach: weights are proportional to the correlation
    between each feature and the target variable. Normalized by feature variance.
    
    Parameters:
    -----------
    dataset : dict
        Dataset dictionary with keys:
        - "X": feature matrix (n_samples x n_features)
        - "y": labels (n_samples,)
    activation : str, optional
        Activation function: 'sigmoid' or 'relu' (default: 'sigmoid')
    
    Returns:
    --------
    dict
        Dictionary containing:
        - "weights": calculated weight vector
        - "bias": bias term (mean of target)
        - "method": weight calculation method
        - "activation": activation function used
        - "correlations": correlation of each feature with target
        - "n_features": number of features
    """
    
    X = dataset["X"]
    y = dataset["y"]
    n_samples, n_features = X.shape
    
    print("=" * 70)
    print("WEIGHT CALCULATION: Pearson Correlation")
    print("=" * 70)
    print("\nDataset Information:")
    print("  - Number of samples: {}".format(n_samples))
    print("  - Number of features: {}".format(n_features))
    print("  - Activation function: {}".format(activation))
    
    # Calculate correlations
    correlations = []
    for i in range(n_features):
        corr = np.corrcoef(X[:, i], y)[0, 1]
        correlations.append(corr)
    
    # Normalize correlations to use as weights
    correlations = np.array(correlations)
    correlations = np.nan_to_num(correlations)  # Handle NaN
    
    # Scale weights
    weights = correlations / (np.sum(np.abs(correlations)) + 1e-8)
    bias = np.mean(y)
    
    print("\nFeature Correlations with Target:")
    for i in range(n_features):
        print("  - Feature {}: {:.4f}".format(i, correlations[i]))
    
    print("\nCalculated Weights:")
    print("  - Weights: {}".format(weights))
    print("  - Bias: {:.4f}".format(bias))
    
    return {
        "weights": weights,
        "bias": bias,
        "method": "Pearson Correlation",
        "activation": activation,
        "correlations": correlations,
        "n_features": n_features
    }


def compare_weight_calculation_methods(dataset, activation='sigmoid'):
    """
    Compare multiple weight calculation methods on the same dataset.
    
    Parameters:
    -----------
    dataset : dict
        Dataset dictionary
    activation : str, optional
        Activation function to use
    
    Returns:
    --------
    dict
        Dictionary containing results from all methods
    """
    
    print("\n\n" + "=" * 70)
    print("COMPARING WEIGHT CALCULATION METHODS")
    print("=" * 70)
    
    results = {}
    
    # Method 1: Random
    print("\n\nMETHOD 1: Random Initialization")
    print("-" * 70)
    results["random"] = calculate_weights_random(dataset, activation)
    
    # Method 2: Hebbian
    print("\n\nMETHOD 2: Hebbian Learning")
    print("-" * 70)
    results["hebbian"] = calculate_weights_hebbian(dataset, activation, learning_rate=0.01, epochs=100)
    
    # Method 3: Gradient Descent
    print("\n\nMETHOD 3: Gradient Descent")
    print("-" * 70)
    results["gradient_descent"] = calculate_weights_gradient_descent(dataset, activation, learning_rate=0.01, epochs=100)
    
    # Method 4: Numerical Gradient
    print("\n\nMETHOD 4: Numerical Gradient + L-BFGS-B")
    print("-" * 70)
    results["numerical_gradient"] = calculate_weights_numerical_gradient(dataset, activation, max_iterations=1000)
    
    # Method 5: Pearson Correlation
    print("\n\nMETHOD 5: Pearson Correlation")
    print("-" * 70)
    results["pearson"] = calculate_weights_pearson_correlation(dataset, activation)
    
    # Summary comparison
    print("\n\n" + "=" * 70)
    print("SUMMARY COMPARISON OF WEIGHT CALCULATION METHODS")
    print("=" * 70)
    
    print("\nWeights Calculated by Each Method:")
    for method_name, result in results.items():
        print("\n{}: {}".format(method_name.upper(), result["method"]))
        print("  - Weights: {}".format(result["weights"][:3]))
        print("  - Bias: {:.4f}".format(result["bias"]))
        if "final_loss" in result:
            print("  - Final Loss: {:.6f}".format(result["final_loss"]))
        if "loss_history" in result:
            print("  - Final Loss: {:.6f}".format(result["loss_history"][-1]))
    
    return results


def print_weights_summary(weights_result):
    """
    Print a summary of weight calculation results.
    
    Parameters:
    -----------
    weights_result : dict
        Weight calculation result dictionary
    """
    
    print("\n" + "=" * 70)
    print("WEIGHTS SUMMARY")
    print("=" * 70)
    print("\nMethod: {}".format(weights_result["method"]))
    print("Activation Function: {}".format(weights_result["activation"]))
    print("\nWeight Vector:")
    for i, w in enumerate(weights_result["weights"]):
        print("  - Weight[{}]: {:.6f}".format(i, w))
    print("\nBias Term: {:.6f}".format(weights_result["bias"]))
    
    if "loss_history" in weights_result:
        losses = weights_result["loss_history"]
        print("\nTraining Loss:")
        print("  - Initial: {:.6f}".format(losses[0]))
        print("  - Final: {:.6f}".format(losses[-1]))
        print("  - Improvement: {:.6f}".format(losses[0] - losses[-1]))


def main():
    """
    Main function demonstrating weight calculation methods.
    """
    print("\n")
    print("#" * 70)
    print("#" + " " * 68 + "#")
    print("#" + "  WEIGHT CALCULATION AND OPTIMIZATION MODULE".center(68) + "#")
    print("#" + " " * 68 + "#")
    print("#" * 70)
    
    print("\n\nGenerating dataset for weight calculation...")
    # Use built-in function instead of importing
    dataset = generate_linearly_separable_data(n_samples=100, n_features=3, random_seed=42)
    
    # Example 1: Single method with Sigmoid
    print("\n\n" + "#" * 70)
    print("EXAMPLE 1: GRADIENT DESCENT WITH SIGMOID ACTIVATION")
    print("#" * 70)
    weights_gd = calculate_weights_gradient_descent(
        dataset, 
        activation='sigmoid', 
        learning_rate=0.05, 
        epochs=200
    )
    print_weights_summary(weights_gd)
    
    # Example 2: Single method with ReLU
    print("\n\n" + "#" * 70)
    print("EXAMPLE 2: GRADIENT DESCENT WITH RELU ACTIVATION")
    print("#" * 70)
    weights_relu = calculate_weights_gradient_descent(
        dataset,
        activation='relu',
        learning_rate=0.05,
        epochs=200
    )
    print_weights_summary(weights_relu)
    
    # Example 3: Pearson Correlation
    print("\n\n" + "#" * 70)
    print("EXAMPLE 3: PEARSON CORRELATION METHOD")
    print("#" * 70)
    weights_pearson = calculate_weights_pearson_correlation(dataset, activation='sigmoid')
    print_weights_summary(weights_pearson)
    
    # Example 4: Compare all methods
    print("\n\n" + "#" * 70)
    print("EXAMPLE 4: COMPARING ALL WEIGHT CALCULATION METHODS")
    print("#" * 70)
    all_results = compare_weight_calculation_methods(dataset, activation='sigmoid')
    
    # Final Summary
    print("\n\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("\nWeight Calculation Methods Available:")
    print("  1. Random Initialization - Baseline random weight initialization")
    print("  2. Hebbian Learning - Classic neural learning rule")
    print("  3. Gradient Descent - Standard optimization with backprop")
    print("  4. Numerical Gradient - Finite differences + L-BFGS-B")
    print("  5. Pearson Correlation - Statistical correlation method")
    print("\nSupported Activation Functions:")
    print("  - sigmoid: (0, 1) output range")
    print("  - relu: [0, inf) output range")
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
