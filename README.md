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

**Householder reflection matrix:** For a nonzero vector $v$, the Householder reflection matrix is 

$$H=I-2\frac{vv^T}{v^Tv}$$

In this implementation, the vector $v$ is 

$$v = x - \lVert x \rVert e_1$$

which gives the result 

$$Hx = \lVert x \rVert e_1$$

where $e_1$ is the first standard basis vector. Due to its norm preserving nature, this property has several applications, for example eigenvalues and QR decomposition.

## Matrices
**Rank:** To calculate the rank of a matrix, it is first brought to Reduced Row Echelon form (RREF) and then the number of non zero rows are counted. 

**QR Decomposition:** This implementation has two QR decompositions, normal QR and Householder QR. Householder QR applies Householder reflections to each subcolumn to zero out entries that allow R to be upper triangular. For example, let $x$ be a subcolumn fo the matrix $A$, then 

$$H_1 x = \lVert x \rVert e_1$$

which has every entry zeroed out except the first. So 

$$A_1 = H_1 A$$

where the first column of $A_1$ is the zeroed out column. Then, to form the second subcolumn, lets say $y$, we form the vector without the first element of the second column and repeat. So finally we obtain 

$$R = H_n H_{n-1} ... H_2 H_1 A$$

and since $$QR = A$$, Q must be 

$$Q = H_1 H_2 ... H_{n-1} H_n$$ 

due to orthogonality of H.

**LU Decomposition:** This implementation always returns 3 matrices, $P$, $L$, $U$, where $P$ is the permutation matrix, $L$ is the lower triangular matrix which always has 1's on its main diagonal, and $U$ is the upper triangular matrix. A form of modified gaussian elimination is done, such that the lower triangular parts and the upper triangular parts are saved in the same matrix, then extracted later. This is feasible since the diagonals of $L$ are always 1. The determinant changes on row switches, so the sign of the determinant is saved in this method and used when the determinant is calculated.

**Hessenberg Matrix:** A Hessenberg matrix is a matrix that is nearly triangular and a matrix $H$ obtained from an original matrix $A$ such that 

$$A = QHQ^T$$

To obtain a nearly triangular matrix, we take the entries below the first subdiagonal $x$ and obtain 

$$Hx = \lVert x \rVert e_1$$

by using Householder reflections. We then embed this into an identity matrix with the same size as the original matrix. Then, the following similarity transformation is applied

$$A_1 = Q_1^T A Q_1$$

Where $Q_1$$ is the embedded Householder reflection, and $A_1$ is the matrix with all the elements under the column of the subdiagonal zeroed out. Since householder is symmetric and orthogonal, this simplifies to

$$A_1 = Q_1 A Q_1$$

This process is applied iteratively until all the elements under the subdiagonal are zeroed out. Due to its similarity transformations, this matrix is used to calculate Eigenvalues to reduce time complexity. 

**Inverse:** In this implementation, to obtain the inverse of a matrix, the identity matrix is partitioned onto it. Then, the original part of the matrix is brought to Reduced Row Echelon Form, and by this process, the identity matrix which was earlier partitioned has transformed into the inverse of the original matrix. 

**Determinant:** To calculate the determinant, we use the property of triangular matrices and the fact that the determinant of a diagonal matrix is the product of the diagonal elements. To take advantage of this property, LU decomposition of the original matrix is performed, and the diagonal elements of the upper triangular matrix $U$ are multiplied. The $L$ matrix can be ignored because it only has 1's on its diagonals. To account for row switches, we the multiply by the sign of the determinant obtained in the LU decomposition, which may be either 1 or -1.

**Eigenvalues:** In this implementation, the eigenvalues are computed using a QR algorithm as follows

$$A_k = Q_k R_k$$
$$A_{k+1} = Q_k^T A_k Q_k = R_k Q_k$$

Where the diagonal elements of $A$ eventually converge to the eigenvalues of the original matrix. The intuitive explanation of why this occurs is because, QR iteration repeatedly changes coordinates of the vectors into a basis that becomes more aligned with the eigenvectors. So as $Q_k$ approaches the eigenvector matrix $V$, then 

$$Q_k^T A Q_k$$

approaches

$$V^{-1} A V = D$$

To reduce computational complexity, the original matrix is first converted into a Hessenberg matrix. This does not change the eigenvalues since Hessenberg Matrices are formed via similarity transformations. To make convergence faster, a Wilkinson shift is included as follows

$$A_k - \mu I = Q_k R_k$$
$$A_{k+1} = R_k Q_k + \mu I$$

where $I$ is the corresponding identity matrix, and $\mu$ is the eigenvalue of the bottom left 2x2 submatrix of the original matrix. This makes convergence faster because $A - \mu I$ becomes nearly singular causing the QR algorithm to make the subdiagonal entries zero out faster.

# Limitations
- Floating point arithmetic
- Complex eigenvalues are not supported
- QR algorithm doesn't converge for large matrices

















