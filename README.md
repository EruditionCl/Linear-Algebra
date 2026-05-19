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
\color{red}{7} & \color{blue}{4} & \color{teal}{8} \\
\color{blue}{5} & \color{red}{7} & \color{purple}{3} \\
\color{teal}{7} & \color{purple}{8} & \color{red}{5}
\end{bmatrix} 
$$

Which is stored as a nested list [[7, 4, 8], [5, 7, 3], [7, 8, 5]], and lets say the operation to be performed is the transpose operation, so we obtain:

$$
\begin{bmatrix}
\color{red}{7} & \color{blue}{5} & \color{teal}{7} \\
\color{blue}{4} & \color{red}{7} & \color{purple}{8} \\
\color{teal}{8} & \color{purple}{3} & \color{red}{5}
\end{bmatrix} 
$$

When this operation occurs, the original nested list is modified to return [[7, 5, 7], [4, 7, 8], [8, 3, 5]].

# Mathematical Background

This section explains how some linear algebra concepts have been implemented, as due to the nature of Python some mathematical concepts could not be directly translated into Python, and another method must be used. This section displays the alternative methods used.

## Vectors
**Linear Independence:** To check linear dependency of $n$ vectors, the vectors are placed into a matrix as rows, and whether its linearly independent or not is dependent on whether the rank of the matrix is equal to the number of rows.

**Span:** To check whether $n$ vectors span $\mathbb{R}^n$, the vectors are placed into matrix as rows, and if the rank of the matrix is equal to the number of columns, they span $\mathbb{R}^n$

**Householder reflection matrix:** For a nonzero vector $v$, the Householder reflection matrix is $$H=I-2\frac{vv^T}{v^Tv}$$. In this implementation, the vector $v$ is $$v = x - \lVert x \rVert e_1$$, which gives the rsult $$Hx = \lVert x \rVert e_1$$, where $e_1$ is the first standard basis vector. Due to its norm preserving nature, this property has several applications, for example eigenvalues and QR decomposition.

## Matrices
**Rank:** To calculate the rank of a matrix, it is first brought to Reduced Row Echelon form (RREF) and then the number of non zero rows are counted. 

**QR Decomposition:** This implementation has two QR decompositions, normal QR and Householder QR. Householder QR applies Householder reflections to each subcolumn to zero out entries that allow R to be upper triangular. For example, let $x$ be a subcolumn fo the matrix $A$, then $$H_1 x = \lVert x \rVert e_1$$, which has every entry zeroed out except the first. So $$A_1 = H_1 A$$, where the first column of $A_1$ is the zeroed out column. Then, to form the second subcolumn, lets say $y$, we form the vector without the first element of the second column and repeat. We even














