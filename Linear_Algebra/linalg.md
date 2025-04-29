# Linear Algebra Essential Concepts

## Vector Spaces
### Properties
- **Closed under addition and scalar multiplication**
- **Zero vector**: Additive identity
- **Inverse**: Every vector has an additive inverse
- **Associativity and commutativity** of addition
- **Distributive properties** of scalar multiplication

## Linear Independence
- Vectors $\{v_1, ..., v_n\}$ are linearly independent if:
  - $c_1v_1 + c_2v_2 + ... + c_nv_n = 0$ implies all $c_i = 0$
- **Basis**: Linearly independent set that spans the space
- **Dimension**: Number of vectors in a basis

## Matrix Theory
### Types of Matrices
1. **Square Matrix**: $n \times n$ dimensions
2. **Symmetric Matrix**: $A = A^T$
3. **Orthogonal Matrix**: $AA^T = I$
4. **Positive Definite Matrix**: All eigenvalues > 0

### Matrix Operations
- **Determinant**: $det(A)$ or $|A|$
- **Inverse**: $AA^{-1} = I$
- **Rank**: Dimension of column/row space
- **Trace**: Sum of diagonal elements

## Linear Transformations
### Key Concepts
1. **Definition**: $T: V \rightarrow W$ where:
   - $T(u + v) = T(u) + T(v)$
   - $T(cv) = cT(v)$
2. **Kernel**: $ker(T) = \{v \in V : T(v) = 0\}$
3. **Image**: $im(T) = \{T(v) : v \in V\}$

## Eigenvalues and Eigenvectors
### Definition
For matrix $A$:
- $Av = \lambda v$
- $v$ is eigenvector
- $\lambda$ is eigenvalue

### Properties
1. **Characteristic Equation**: $det(A - \lambda I) = 0$
2. **Diagonalization**: $A = PDP^{-1}$ where:
   - $D$ is diagonal matrix of eigenvalues
   - $P$ is matrix of eigenvectors

## Applications
- **Principal Component Analysis (PCA)**
- **Singular Value Decomposition (SVD)**
- **Linear Least Squares**
- **Matrix Factorization**