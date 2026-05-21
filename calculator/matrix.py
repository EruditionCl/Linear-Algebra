
import copy
import math
from .exceptions import *
from .utils import *


class Matrix:
    def __init__(self, *args):

        from .vector import Vector

        """Initializes a Matrix.

        Args:
            *args: Rows as vectors or lists.
        """
        if all(isinstance(arg, (list, tuple)) for arg in args):
            vectors = [Vector(*arg) for arg in args]
            vector_checker(*vectors)
            self.rowspace = vectors

        else:
            vector_checker(*args)
            self.rowspace = [arg for arg in args]
                

    @property
    def rows(self):
        """Number of rows.

        Returns:
            int: The number of rows.
        """
        return len(self.rowspace)   
    
    @property 
    def columns(self):
        """Number of columns.

        Returns:
            int: The number of columns.
        """
        return len(self.rowspace[0])
    
    def rank(self):
        """The rank of the matrix.

        Returns:
            int: The rank.
        """
        return sum(1 for row in self.gauss_jordan() if any(val != 0 for val in row))
    
    def nullity(self):
        """The nullity of the matrix.

        Returns:
            int: The nullity.
        """
        return self.columns - self.rank()
    
    def is_square(self):
        """Checks if the matrix is square.

        Returns:
            bool: True if square.
        """
        return self.rows == self.columns
    
    def is_triangular(self):
        """Checks if the matrix is triangular.

        Returns:
            bool: True if triangular.
        """
        return self.is_uppertriangular() or self.is_lowertriangular()

    def is_uppertriangular(self):
        """Checks if the matrix is an upper triangular matrix.

        Returns:
            bool: True if upper triangular.
        """
        return self.is_square() and all(is_zero(self[i][j])
                   for i in range(self.rows) 
                   for j in range(self.rows) 
                   if i > j)
    
    def is_lowertriangular(self):
        """Checks if the matrix is a lower triangular matrix.

        Returns:
            bool: bool: True if lower triangular.
        """
        return self.is_square() and all(is_zero(self[i][j])
                   for i in range(self.rows) 
                   for j in range(self.rows) 
                   if i < j)
    
    def is_symmetric(self):
        """Checks if the matrix is symmetric.

        Returns:
            bool: True if symmetric.
        """
        return self.is_square() and all(is_approx(self[i][j], self[j][i]) 
                                        for i in range(self.rows) 
                                        for j in range(self.rows))
    
    def stabilize(self, eps = EPS):
        """Creates a stabilized copy of the matrix where extremely small values are rounded to exactly zero.

        Args:
            eps (float, optional): The tolerance threshold. Defaults to EPS.

        Returns:
            Matrix: A stabilized copy of the matrix.
        """
        A = copy.deepcopy(self)
        for i in range(A.rows):
            for j in range(A.columns):
                if is_zero(A[i][j], eps):
                    A[i][j] = 0
        return A
    
    def transpose(self):

        from .vector import Vector

        """Computes the transpose of the matrix.

        Returns:
            Matrix: A new matrix with rows and columns swapped.
        """
        return Matrix(*[Vector(*[self[i][j] 
                for i in range(0, self.rows)]) 
                for j in range(0, self.columns)])   
    
    def gaussian(self, eps=EPS):
        """Performs Gaussian elimination to compute the Row Echelon Form (REF) of the matrix.

        Args:
            eps (float, optional): Tolerance. Defaults to EPS.

        Returns:
            Matrix: The matrix in row echelon form.
        """
        A = copy.deepcopy(self)
        m, n = A.rows, A.columns

        for j in range(min(m, n)):
            pivot_row = max(range(j, m), key=lambda i: abs(A[i][j]))
            col_scale = max(abs(A[i][j]) for i in range(j, m))

            if col_scale == 0 or abs(A[pivot_row][j]) < eps * col_scale:
                continue

            if pivot_row != j:
                A[j], A[pivot_row] = A[pivot_row], A[j]

            pivot = A[j][j]
            if is_zero(pivot, eps): continue 
            A[j] = A[j] / pivot

            for i in range(j + 1, m):
                factor = A[i][j]
                if is_zero(factor, eps):
                    A[i][j] = 0.0
                    continue
                A[i] = A[i] - factor * A[j]

        A = A.stabilize()
        return A
    
    def gauss_jordan(self, eps=EPS):
        """Performs Gauss-Jordan elimination to compute the Reduced Row Echelon Form (RREF) of the matrix.

        Args:
            eps (float, optional): Tolerance. Defaults to EPS.

        Returns:
            Matrix: The matrix in reduced row echelon form.
        """
        A = copy.deepcopy(self).gaussian(eps=eps) 
        
        for j in range(min(A.rows, A.columns)):
            for i in range(A.rows):
                if i != j and is_nonzero(A[i][j], eps): 
                    A[i] = A[i] - A[i][j] * A[j]

        A = A.stabilize()
        return A

    def partition(self, other):
        """Horizontally concatenates another matrix to the right of this matrix.

        Args:
            other (Matrix): The matrix to append.

        Returns:
            Matrix: A new augmented matrix.

        Raises:
            DimensionError: If the number of rows do not match.
            TypeError: If the object passed is not a Matrix.
        """
        if isinstance(other, Matrix):
            if not self.rows == other.rows:
                raise DimensionError("partition() only accepts matrices with equal rows")
            return Matrix(*[(a.partition(b)) 
                            for a, b in zip(self.rowspace, other.rowspace)])
        else:
            raise TypeError("Matrix can only be partitioned by Matrix class")
        
    def homogeneous(self, eps=1e-4):

        from .vector import Vector

        """Solves the homogeneous system of linear equations (Ax = 0).

        Args:
            eps (float, optional): Tolerance. Defaults to 1e-4.

        Returns:
            list[Vector]: A list of vectors representing the solution of the homogeneous system.
        """
        A = copy.deepcopy(self)
        A = A.gauss_jordan(eps=eps) 
        m, n = A.rows, A.columns

        A = A.stabilize()
            
        pivot_map = [-1 for _ in range(n)]
        for i in range(m):
            for j in range(n):
                if is_nonzero(A[i][j], eps):
                    if all(is_zero(A[i][k], eps) for k in range(j)):
                        pivot_map[j] = i
                        break

        solution = []

        for j in range(n):
            if pivot_map[j] == -1: 
                v = Vector(*[0 for _ in range(n)])
                v[j] = 1.0 
                
                for k in range(n):
                    if pivot_map[k] != -1: 
                        r = pivot_map[k]
                        v[k] = -A[r][j] 
                
                solution.append(v)

        return solution

    def qr(self):

        from .vector import Vector

        """Computes the QR decomposition of the matrix using Gram-Schmidt orthonormalization.

        Returns:
            tuple[Matrix, Matrix]: A tuple (Q, R).
            Q is an orthogonal matrix.
            R is an upper triangular matrix.
        """
        A = copy.deepcopy(self)
        Q = Matrix(*Vector.gram_schmidt_orthonormal(*A.transpose().rowspace)).transpose()
        R = Q.transpose() @ A

        Q, R = Q.stabilize(), R.stabilize()
        return Q, R
    
    def qr_householder(self):

        from .vector import Vector

        """Computes the QR decomposition of the matrix using Householder reflections.

        Returns:
            tuple[Matrix, Matrix]: A tuple (Q, R.
            Q is an orthogonal matrix.
            R is an upper triangular matrix.
        """
        A = copy.deepcopy(self)
        m, n = A.rows, A.columns
        Q = Matrix.identity(m)

        for k in range(min(m, n)):
            x = Vector(*[A[i][k] for i in range(k, m)])
            H_ = x.householder()
            H = Matrix.identity(m)

            for i in range(H_.rows):
                for j in range(H_.rows):
                    H[k + i][k + j] = H_[i][j]

            A = H @ A
            Q = Q @ H

        R = A
        Q, R = Q.stabilize(), R.stabilize()
        return Q, R
        
    @must_be_square
    def plu(self):
        """Computes the PLU decomposition of the square matrix.

        Returns:
            tuple[Matrix, Matrix, Matrix]: A tuple (P, L, U).
            P is a permutation matrix.
            L is a lower triangular matrix. 
            U is an upper triangular matrix.
        """
        n = self.rows
        P = Matrix.identity(n)
        A = copy.deepcopy(self)
        det_sign = 1

        for j in range(0, n):
            if is_zero(A[j][j]):
                for k in range(j + 1, n):
                    if is_nonzero(A[k][j]):
                        A[j], A[k] = A[k], A[j]
                        P[j], P[k] = P[k], P[j]
                        det_sign *= -1
                        break

            if is_nonzero(A[j][j]):
                for i in range(j + 1, n):
                    A[i][j] = A[i][j] / A[j][j]
                    for k in range(j + 1, n):
                        A[i][k] -= (A[i][j]) * A[j][k]

        L = Matrix.identity(n)
        U = Matrix.identity(n)

        for j in range(n):
            for i in range(n):
                if i < j:
                    L[j][i] = A[j][i]
                elif i >= j:
                    U[j][i] = A[j][i]
                
        self._det_sign = det_sign

        L, U = L.stabilize(), U.stabilize()
        return P, L, U
    
    @must_be_square
    def hessenberg(self):

        from .vector import Vector

        """Reduces the square matrix to upper Hessenberg form using Householder reflections.

        Returns:
            Matrix: The matrix in upper Hessenberg form.
        """
        n = self.rows
        A = copy.deepcopy(self)

        for k in range(n - 2):
            x = Vector(*[A[i][k] for i in range(k + 1, n)])
            H = x.householder()
            I = Matrix.identity(n)

            for i in range(H.rows):
                for j in range(H.rows):
                    I[k + 1 + i][k + 1 + j] = H[i][j]
            
            A = I @ A @ I

        
        A = A.stabilize()
        return A
        
    @must_be_square
    def inverse(self):

        from .vector import Vector

        """Computes the inverse of the square matrix using Gauss-Jordan elimination.

        Returns:
            Matrix: The inverse of the matrix.

        Raises:
            UninvertibleMatrixError: If the matrix is singular.
        """
        n = self.rows
        if self.gauss_jordan()[n - 1] == Vector.zero_vector(n):
            raise UninvertibleMatrixError()
        
        A = self.partition(Matrix.identity(n)).gauss_jordan()
        return Matrix(*(Vector(*(A[i][j] 
                                for j in range(n, 2*n))) 
                                for i in range(n)))

    @must_be_square
    def trace(self):
        """Computes the trace of the square matrix.

        Returns:
            float: The trace of the matrix.
        """
        return sum(self[i][i] for i in range(0, self.rows))

    @must_be_square
    def determinant(self):
        """Computes the determinant of the square matrix using PLU decomposition.

        Returns:
            float: The determinant of the matrix.
        """
        P, L, U = self.plu()
        return self._det_sign * math.prod(U[i][i] for i in range(self.rows))
    
    @must_be_square
    def eigenvalues(self):
        """Computes the eigenvalues of the square matrix.

        Returns:
            list[float]: A list of eigenvalues sorted in descending order.

        Raises:
            ValueError: If the matrix contains complex eigenvalues.
        """
        A = copy.deepcopy(self)
        n = A.rows
        I = Matrix.identity(n)

        if not A.is_triangular():
        
            A = A.hessenberg()

            while any(is_nonzero(A[i][i-1]) for i in range(1, n)):

                a = A[n-2][n-2]
                b = A[n-2][n-1]
                c = A[n-1][n-2]
                d = A[n-1][n-1]

                delta = (a - d) / 2
                discriminant = delta ** 2 + b * c
                
                if discriminant < 0:
                    raise ValueError("Imaginary Eigenvalues")
                
                denominator = abs(delta) + pow(discriminant, 0.5)

                if is_zero(denominator):
                    mu = d 
                else:
                    mu = d - ((sign(delta) * b * c) / denominator)
                # Wilkinson shift

                A = A - mu * I
                Q, R= A.qr_householder()
                A = R @ Q + (mu * I)

        
        result = [A[i][i] for i in range(n)]
        result.sort(reverse=True)

        return result


    def eigenvectors(self):
        """Extracts all eigenvectors from the calculated eigenpairs.

        Returns:
            list[Vector]: A flat list of all eigenvectors associated with the matrix.
        """
        A = copy.deepcopy(self)
        eigenpairs = A.eigenpairs()

        eigenvectors = [v for value in eigenpairs.values() 
                        for v in (value if isinstance(value, list) 
                                  else [value])]

        return eigenvectors

    
    def eigenpairs(self):
        """Computes the eigenvalues and their corresponding eigenvectors.

        Returns:
            dict: A dictionary mapping eigenvalues to lists of corresponding eigenvectors, 
            sorted in descending order of the eigenvalue.
        """
        A = copy.deepcopy(self)
        I = Matrix.identity(A.rows)
        eigenvalues = A.eigenvalues()

        eigenpairs = {}
        for eigenvalue in eigenvalues:
            B = A - (eigenvalue * I)
            eigenpairs[eigenvalue] = B.homogeneous(eps=1e-4)

        eigenpairs = {k: eigenpairs[k] for k in sorted(eigenpairs, reverse = True)}

        return eigenpairs


    def diagonalization(self):
        """Computes the diagonalization of the square matrix.

        Returns:
            tuple[Matrix, Matrix, Matrix]: A tuple (P, D, P_inverse). 
            P is the eigenvector matrix, 
            D is the diagonal eigenvalue matrix, 
            P_inverse is the inverse of the eigenvector matrix.

        Raises:
            UndiagonalizableError: If the matrix does not have 'n' linearly independent eigenvectors.
        """
        A = copy.deepcopy(self)
        I = Matrix.identity(A.columns)
        eigenpairs = A.eigenpairs()

        aligned_eigenvalues = []
        aligned_eigenvectors = []

        for eigenvalue, vectors in eigenpairs.items():
            for v in vectors:
                aligned_eigenvalues.append(eigenvalue)
                aligned_eigenvectors.append(v)

        if Matrix(*aligned_eigenvectors).rank() != A.columns:
            raise UndiagonalizableError("Matrix doesn't have " \
            "n linear independent eigenvectors")
            
        P = Matrix(*aligned_eigenvectors).transpose()
        D = I

        for i in range(A.columns):
            D[i][i] = aligned_eigenvalues[i]

        P, D = P.stabilize(), D.stabilize()
        return P, D, P.inverse()
    
    def orthogonal_diagonalization(self):
        """Computes the orthogonal diagonalization of a symmetric matrix.

        Returns:
            tuple[Matrix, Matrix, Matrix]: A tuple (Q, D, Q_inverse).
            Q is an orthogonal matrix of eigenvectors.
            D is a diagonal eigenvalue matrix.
            Q_inverse is the inverse of the orthogonal eigenvector matrix.

        Raises:
            UndiagonalizableError: If the matrix is not symmetric.
        """
        if not self.is_symmetric():
            raise UndiagonalizableError("Matrix must be symmetric")
        
        A = copy.deepcopy(self)
        P, D, P_1 = A.diagonalization()
        Q = Matrix(*[column.normalize() for column in P.transpose()]).transpose()
        
        return Q, D, Q.inverse()


    def svd(self):

        from .vector import Vector

        """Computes the Singular Value Decomposition of the matrix.

        Returns:
            tuple[Matrix, Matrix, Matrix]: A tuple (U, Sigma, V_T). 
            U is the left singular vectors.
            Sigma is the diagonal singular values matrix.
            V_T is the transposed right singular vectors.
        """
        A = copy.deepcopy(self)
        ATA = A.transpose() @ A
        V, D = ATA.orthogonal_diagonalization()[0:2]
        singular = [math.sqrt(abs(D[i][i])) for i in range(D.rows)]

        u_list = []
        non_zero_singulars = []
        v_T_list = [] 

        for i in range(len(singular)):
            if is_zero(singular[i]): 
                continue
                    
            u = A @ Matrix(V.transpose()[i]).transpose()
            u = u.transpose()[0]
                
            if is_zero(u.norm): 
                continue
                    
            u = u.normalize()
                
            u_list.append(u)
            non_zero_singulars.append(singular[i])
            v_T_list.append(V.transpose()[i])
            
            r = len(u_list)
            U = Matrix(*u_list).transpose()

            Sigma = Matrix(*[
                        Vector(*[non_zero_singulars[i] 
                                if i == j else 0 
                                for j in range(r)]) 
                                for i in range(r)])
            
            V_T = Matrix(*v_T_list)

            return U, Sigma, V_T
    
    def pseudoinverse(self):
        """Computes the Moore-Penrose pseudoinverse of the matrix.

        Returns:
            Matrix: The pseudoinverse of the matrix.
        """
        A = copy.deepcopy(self)

    def __eq__(self, other):
        """Checks equality with another matrix.

        Args:
            other: The other matrix.

        Returns:
            bool: True if equal.
        """
        if isinstance(other, Matrix):
            return all(is_approx(a, b) for a, b in zip(self.rowspace, other.rowspace))
        else:
            return self == other
        
    def __round__(self, ndigits = 12):
        """Rounds the matrix entries.

        Args:
            ndigits (int): Number of digits.

        Returns:
            Matrix: The rounded matrix.
        """
        return Matrix(*[round(row, ndigits) for row in self.rowspace])
    
    def __str__(self):
        """String representation of the matrix.

        Returns:
            str: The string.
        """
        return '\n'.join(str(row) for row in self.rowspace)
    
    def __getitem__(self, key):
        """Gets a row by index.

        Args:
            key: The index.

        Returns:
            Vector: The row.
        """
        return self.rowspace[key]
    
    def __setitem__(self, key, value):
        """Sets a row by index.

        Args:
            key: The index.
            value: The row.

        Returns:
            Matrix: Self.
        """
        self.rowspace[key] = value
        return self
    
    def __delitem__(self, key):
        """Deletes a row by index.

        Args:
            key: The index.

        Returns:
            Matrix: Self.
        """
        del self.rowspace[key]
        return self
    
    def __neg__(self):
        """Negates the matrix.

        Returns:
            Matrix: The negated matrix.
        """
        return Matrix(*(-a for a in self.rowspace))
    
    def __pos__(self):
        """Positive operator.

        Returns:
            Matrix: Self.
        """
        return self
    
    def __add__(self, other):
        """Adds two matrices.

        Args:
            other (Matrix): The other matrix.

        Returns:
            Matrix: The sum.
        """
        validate_matrix_compatibility(self, other)
        return Matrix(*(a + b for a, b in zip(self.rowspace, other.rowspace)))
    
    def __sub__(self, other):
        """Subtracts another matrix.

        Args:
            other (Matrix): The other matrix.

        Returns:
            Matrix: The difference.
        """
        return self + -other
    
    def __mul__(self, other):

        from .vector import Vector

        """Multiplies by a scalar.

        Args:
            other: The scalar.

        Returns:
            Matrix: The scaled matrix.

        Raises:
            TypeError: If other is not a scalar.
            NotImplementedError: For unsupported types.
        """
        if isinstance(other, (Matrix, Vector)):
            raise TypeError("__mul__ only accepts scalars.")
        elif isinstance(other, (int, float)):
            return Matrix(*(other * a for a in self.rowspace))
        else:
            raise NotImplementedError(f"{type(other)} is not accounted for.")

    def __matmul__(self, other):

        from .vector import Vector

        """Multiplies by a matrix.

        Args:
            other (Matrix): The other matrix.

        Returns:
            Matrix: The product.

        Raises:
            TypeError: If other is not a Matrix.
        """
        if isinstance(other, Matrix):
            validate_matrix_mult_compatibility(self, other)
            return round(Matrix(*[Vector(*[self[j] @ other.transpose()[i] 
                 for j in range(0, self.rows)]) 
                 for i in range(0, other.columns)]).transpose().stabilize())
        else:
            raise TypeError("__matmul__ accepts only matrices.")
        
    def __rmatmul__(self, other):
        """Right matrix multiplication.

        Raises:
            TypeError: Always.
        """
        raise TypeError("Matrix Multiplication without a matrix is not meaningful")

    def __rmul__(self, other):
        """Right multiplication by scalar.

        Args:
            other: The scalar.

        Returns:
            Matrix: The scaled matrix.
        """
        return self * other
    
    def __truediv__(self, other):

        from .vector import Vector

        """Divides by a scalar.

        Args:
            other: The scalar.

        Returns:
            Matrix: The divided matrix.

        Raises:
            TypeError: If other is not a scalar.
            NotImplementedError: For unsupported types.
        """
        if isinstance(other, (Vector, Matrix)):
            raise TypeError("__truediv__ only accepts scalars")
        elif isinstance(other, (int, float)):
            return Matrix(*(a / other for a in self.rowspace))
        else:
            NotImplementedError(f"{type(other)} is not accounted for.")

    def __rtruediv__(self, other):
        """Right division.

        Raises:
            TypeError: Always.
        """
        raise TypeError("Division by Matrix is not meaningful")
    
    @staticmethod
    def solve(A, b):

        from .vector import Vector

        """Solves the system of linear equations Ax = b.

        Args:
            A (Matrix): The coefficient matrix.
            b (Vector): The constants vector.

        Returns:
            Vector or list[Vector]: A single Vector solution if a unique solution exists,
                or a list of vectors mapping the solution space if infinite solutions exist.

        Raises:
            TypeError: If inputs are not of correct types.
            InconsistentSystemError: If the system has no solution.
        """
        if not isinstance(A, Matrix):
            raise TypeError("A must be a Matrix")
        elif not isinstance(b, Vector):
            raise TypeError("b must be a Vector")
        elif b == Vector.zero_vector(b.dimension):
            return A.homogeneous()
        
        C = A.partition(Matrix(b).transpose()).gauss_jordan()

        for row in C:
            if all(is_zero(x) for x in row[:-1]) and is_nonzero(row[-1]):
                raise InconsistentSystemError()
            
        if A.rank() < A.columns:
            C = A.partition(Matrix(-b).transpose())
            vectors = C.homogeneous()
            vectors.reverse()
            for vector in vectors:
                del vector[-1]
            return vectors
        
        x = Vector(*[C[i][C.columns - 1] for i in range(C.rows)])
        return x
    
    @staticmethod
    def least_squares(A, b):

        from .vector import Vector

        """Computes the least squares solution to an overdetermined system Ax = b.

        Args:
            A (Matrix): The coefficient matrix.
            b (Vector): The target vector.

        Returns:
            Vector or list[Vector]: The closest approximation vector(s) x solving the normal equations.

        Raises:
            TypeError: If inputs are of invalid types.
        """
        if not isinstance(A, Matrix):
            raise TypeError("A must be a Matrix")
        elif not isinstance(b, Vector):
            raise TypeError("b must be a Vector")
        
        AT = A.transpose()
        normal = A.transpose() @ A
        B = AT @ Matrix(b).transpose()
        B = B.transpose()

        return Matrix.solve(normal, *B)
    
    @staticmethod
    def least_squares_error(A, b):
        """Computes the minimal distance (error) for the least squares approximation of Ax = b.

        Args:
            A (Matrix): The coefficient matrix.
            b (Vector): The target vector.

        Returns:
            float: The Euclidean norm of the error vector.
        """
        x = Matrix.least_squares(A, b)[0]
        Ax = (A @ Matrix(x).transpose()).transpose()[0]
        error = b - Ax

        return error.norm
            
    @staticmethod
    def identity(n):

        from .vector import Vector

        """Creates an n x n identity matrix.

        Args:
            n (int): The dimension of the identity matrix.

        Returns:
            Matrix: An n x n identity matrix.
        """
        return Matrix(*[Vector(*[1 if i == j else 0 
                                 for j in range(n)]) 
                                 for i in range(n)])
    
    @staticmethod
    def zero_matrix(m, n):

        from .vector import Vector

        """Creates an m x n zero matrix.

        Args:
            m (int): Number of rows.
            n (int): Number of columns.

        Returns:
            Matrix: An m x n matrix containing entirely zeros.
        """
        return Matrix(*[Vector.zero_vector(n) for _ in range(m)])
