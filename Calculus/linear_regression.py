import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional

class LinearRegression:

    def __init__(self, learning_rate: float = 0.01, n_iterations: int = 1000):
        """
        Initialize the linear regression model.
        """
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.cost_history = []
        
    def initialize_parameters(self, n_features: int):
        """
        Initialize model parameters (weights and bias).
        """
        self.weights = np.zeros(n_features)
        self.bias = 0
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the current model parameters.
        """
        return np.dot(X, self.weights) + self.bias
    
    def compute_cost(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute the Mean Squared Error cost function.
        """
        n_samples = X.shape[0]
        y_pred = self.predict(X)
        cost = (1 / (2 * n_samples)) * np.sum((y_pred - y) ** 2)
        return cost
    
    def compute_gradients(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Compute gradients of the cost function with respect to weights and bias.
        """
        n_samples = X.shape[0]
        y_pred = self.predict(X)
        errors = y_pred - y
        
        # Calculate gradients (partial derivatives)
        dw = (1 / n_samples) * np.dot(X.T, errors)  # Gradient for weights
        db = (1 / n_samples) * np.sum(errors)       # Gradient for bias
        
        return dw, db
    
    def update_parameters(self, dw: np.ndarray, db: float):
        """
        Update model parameters using the computed gradients.
        """
        self.weights = self.weights - self.learning_rate * dw
        self.bias = self.bias - self.learning_rate * db
    
    def fit(self, X: np.ndarray, y: np.ndarray, verbose: bool = True) -> List[float]:
        """
        Train the linear regression model using gradient descent.
        """
        # Initialize parameters
        n_features = X.shape[1]
        self.initialize_parameters(n_features)
        self.cost_history = []
        
        # Gradient descent iterations
        for i in range(self.n_iterations):
            # Forward pass - compute predictions and cost
            cost = self.compute_cost(X, y)
            self.cost_history.append(cost)
            
            # Backward pass - compute gradients
            dw, db = self.compute_gradients(X, y)
            
            # Update parameters using gradients
            self.update_parameters(dw, db)
            
            # Print progress
            if verbose and i % 100 == 0:
                print(f"Iteration {i}: Cost = {cost:.6f}")
                
        return self.cost_history
    
    def plot_cost_history(self):
        """Plot the cost history during training."""
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(self.cost_history)), self.cost_history)
        plt.title('Cost History')
        plt.xlabel('Iteration')
        plt.ylabel('Cost (MSE)')
        plt.grid(True)
        plt.show()


# Example usage
def generate_data(n_samples: int = 100, n_features: int = 1, 
                  noise: float = 10.0, seed: Optional[int] = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic data for linear regression.
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Generate true weights and bias
    true_weights = np.random.randn(n_features)
    true_bias = np.random.randn()
    
    # Generate features
    X = np.random.randn(n_samples, n_features)
    
    # Generate target values with noise
    y = np.dot(X, true_weights) + true_bias + np.random.normal(0, noise, n_samples)
    
    return X, y


def main():
    # Generate synthetic data
    X, y = generate_data(n_samples=100, n_features=1, noise=5.0, seed=42)
    
    # Create and train the model
    model = LinearRegression(learning_rate=0.01, n_iterations=1000)
    cost_history = model.fit(X, y)
    
    # Plot the cost history
    model.plot_cost_history()
    
    # For 1D data, visualize the regression line
    if X.shape[1] == 1:
        plt.figure(figsize=(10, 6))
        plt.scatter(X[:, 0], y, label='Data Points')
        
        # Sort X for smooth line plotting
        X_sort = np.sort(X, axis=0)
        y_pred = model.predict(X_sort)
        
        plt.plot(X_sort[:, 0], y_pred, 'r-', linewidth=2, label='Regression Line')
        plt.title('Linear Regression')
        plt.xlabel('X')
        plt.ylabel('y')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    # Print the learned parameters
    print(f"Learned weights: {model.weights}")
    print(f"Learned bias: {model.bias}")


if __name__ == "__main__":
    main()