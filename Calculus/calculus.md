# Differentiability and Optimization

In machine learning, especially in algorithms like linear regression, **differentiability** is crucial for optimization. The cost function (e.g., Mean Squared Error, MSE) must be differentiable so we can compute **gradients**—partial derivatives with respect to each parameter (weights and bias). These gradients indicate the direction and rate of the steepest ascent; by moving in the negative gradient direction (**gradient descent**), we iteratively minimize the cost.

```python
dw = (1 / n_samples) * np.dot(X.T, errors)  # ∂Cost/∂weights
db = (1 / n_samples) * np.sum(errors)       # ∂Cost/∂bias
```
Here, the gradients are calculated using partial derivatives, and parameters are updated accordingly.

---

## Relaxation of Non-Smooth Functions

Many real-world problems involve discrete or non-smooth objectives (e.g., 0-1 loss in classification). These are hard to optimize directly because their gradients are zero or undefined almost everywhere. To address this, we use **smooth approximations**, which are differentiable and provide meaningful gradients for optimization.

---

## Chain Rule and Jacobians

When models become more complex (e.g., neural networks), the **chain rule** allows us to compute gradients through compositions of functions (backpropagation). The **Jacobian matrix** generalizes the derivative for vector-valued functions, essential for understanding how changes in inputs affect outputs in multivariate models.

---

## Practical Impact

- **Convexity:** The MSE cost in linear regression is convex, guaranteeing a single global minimum; gradient descent will always converge if the learning rate is appropriate.
- **Gradients in Iterative Methods:** Gradients guide each step in parameter space, ensuring efficient convergence.
- **Smooth Approximations:** Enable the use of gradient-based methods in otherwise intractable problems.

---

In conclusion, **differentiability** and **gradients** are foundational for training machine learning models. By ensuring our cost functions are smooth and convex (when possible), and by using partial derivatives and the chain rule, we can apply iterative optimization methods like gradient descent to efficiently learn model parameters.