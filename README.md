# Linear-Algebra-Calculator
This project is a basic linear algebra library that is capable of performing most basic mathematical operations related to linear algebra. This project is similar to data science libraries like NumPy, but built entirely from naive python without dependencies. Its features include:
- Vectors and Matrices
- Vector Operations (Angle, Projections)
- Gram-Schmidt process
- Matrix operations (eg: Matrix Multiplication)
- Gauss-Jordan Elimination
- Determinants
- Inverses
- Eigenvalues and Eigenvectors
- Diagonalization
- Various Matrix Decompositions (QR, SVD)
- Solving systems of equations
---
# Working Principle
When a Matrix or Vector object is created, it consists of lists of values. When an operation occurs, these lists of values are edited such that the result is as if the mathematical operation has been applied onto the original object.

For example, take the matrix:

$$
\begin{bmatrix}
\mathbf{7} & 4 & 8 \\
5 & \mathbf{7} & 3 \\
7 & 8 & \mathbf{5}
\end{bmatrix}^T
=
\begin{bmatrix}
\mathbf{7} & 5 & 7 \\
4 & \mathbf{7} & 8 \\
8 & 3 & \mathbf{5}
\end{bmatrix}
$$

$$
\begin{bmatrix}
\mathbf{7} & \color{red}{4} & \color{blue}{8} \\
\color{red}{5} & \mathbf{7} & \color{green}{3} \\
\color{blue}{7} & \color{green}{8} & \mathbf{5}
\end{bmatrix}^T
=
\begin{bmatrix}
\mathbf{7} & \color{red}{5} & \color{blue}{7} \\
\color{red}{4} & \mathbf{7} & \color{green}{8} \\
\color{blue}{8} & \color{green}{3} & \mathbf{5}
\end{bmatrix}
$$









