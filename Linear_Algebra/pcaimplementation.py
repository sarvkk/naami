import numpy as np
import matplotlib.pyplot as plt

# Generating synthetic 2D data
np.random.seed(123)
mean = [0, 0]
cov = [[3, 1], [1, 2]]  # covariance matrix
X = np.random.multivariate_normal(mean, cov, 200)

# 1. Standardize the data (mean=0)
X_centered = X - np.mean(X, axis=0)

# 2. Compute covariance matrix
cov_matrix = np.cov(X_centered, rowvar=False)

# 3. Compute eigenvalues and eigenvectors
eig_vals, eig_vecs = np.linalg.eigh(cov_matrix)

# 4. Sort eigenvectors by eigenvalues (descending)
sorted_indices = np.argsort(eig_vals)[::-1]
eig_vals = eig_vals[sorted_indices]
eig_vecs = eig_vecs[:, sorted_indices]

# 5. Project data onto the first k principal components (here, k=2)
k = 2
X_pca = np.dot(X_centered, eig_vecs[:, :k])

# 6. Variance explained by each component
explained_variance = eig_vals / np.sum(eig_vals)

print("Eigenvalues:", eig_vals)
print("Explained variance ratio:", explained_variance)
print("Principal components (eigenvectors):\n", eig_vecs)

# Plot original data and principal axes
plt.scatter(X[:, 0], X[:, 1], alpha=0.2)
origin = np.mean(X, axis=0)
for i in range(2):
    vec = eig_vecs[:, i] * np.sqrt(eig_vals[i]) * 2
    plt.arrow(origin[0], origin[1], vec[0], vec[1], 
              color='r' if i == 0 else 'g', width=0.05)
plt.title("PCA: Principal Components")
plt.xlabel("X1")
plt.ylabel("X2")
plt.axis('equal')
plt.show()