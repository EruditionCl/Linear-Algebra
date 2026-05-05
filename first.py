
import math
import copy

def dimension_checker(*args):
    dimensions = [arg.dimension for arg in args]
    if len(set(dimensions)) != 1:
        raise DimensionError("Mismatching Dimensions")

def validate_matrix_compatibility(A, B):
    if not (A.rows, A.columns) == (B.rows, B.columns):
        raise DimensionError("Mismatching Dimensions")

def validate_matrix_mult_compatibility(A, B):
    if not (A.columns) == (B.rows):
        raise DimensionError("Mismatching Dimensions")

def vector_checker(*args):
    if args == None:
        return
    for arg in args:
        if not isinstance(arg, Vector):
            raise TypeError("Only Vector entries accepted")
        dimension_checker(*args)
    
def entry_checker(*args):
    if not all(isinstance(arg, (int,float)) for arg in args):
        raise TypeError("Vector class only accepts int and float")
    

def sign(n):
    return 1 if n >= 0 else -1

def distance(u, v):
    vector_checker(u, v)
    return (u-v).norm
    
def dot(u, v):
    return u.dot(v)

def angle(u, v):
    return u.angle(v)

def proj(u, v):
    return u.proj(v)

def partition(u, v):
    return u.partition(v)

EPS = 1e-10

def is_nonapprox(a, b, eps=EPS):
    return abs(a - b) >= eps * max(1.0, abs(a), abs(b))

def is_approx(a, b, eps=EPS):
    return abs(a - b) < eps * max(1.0, abs(a), abs(b))

def is_zero(x, eps = EPS):
    return abs(x) < eps

def is_nonzero(x, eps = EPS):
    return abs(x) >= eps

def must_be_square(func):
    def inner(self, *args, **kwargs):
        if not self.is_square():
            raise DimensionError("Matrix must be square")
        return func(self, *args, **kwargs)
    return inner




    
class DimensionError(Exception):
    pass

class UninvertibleMatrixError(Exception):
    pass

class InconsistentSystemError(Exception):
    pass

class InfiniteSolutionsError(Exception):
    pass

class BasisVectorsError(Exception):
    pass

class UndiagonalizableError(Exception):
    pass



class Tensor:
    def __init__(self):
        pass

class Vector(Tensor):

    def __init__(self, *args):
        if isinstance(args, (tuple, list)):
            self.values = list(args)
        else:
            self.values = []
            entry_checker(*args)
            for arg in args:
                self.values.append(arg)
        self.dimension = len(self.values)

    @property
    def norm(self):
        return self.dot(self) ** 0.5
    
    def normalize(self):
        return Vector(*(a / self.norm for a in self.values))
    
    def dot(self, other):
        dimension_checker(self, other)
        return round(sum((a * b for a, b in zip(self.values, other.values))), 10)
    
    def angle(self, other):
        dimension_checker(self, other)
        return math.acos(round((self.dot(other)) / (self.norm * other.norm)))
    
    def proj(self, other):
        dimension_checker(self, other)
        if is_zero((other.norm ** 2)):
            return Vector.zero_vector(self.dimension)
        return (self.dot(other)/(pow(other.norm, 2))) * other
    
    def orthogonal_decomp(self, other):
        return self.proj(other), self - self.proj(other)
    
    def partition(self, other):
        if isinstance(other, Vector):
            return Vector(*[*self.values, *other.values])
        else:
            raise TypeError("Vector can only be partitioned by Vector class")
    
    def coordinates(self, *args):
        if not Vector.basis(*args): raise BasisVectorsError("*args must form a basis")

        A = Matrix(*args).transpose()
        return Matrix.solve(A, self)
    
    def change_of_basis(self, B, B_prime):
        if not Vector.basis(*B):
            raise BasisVectorsError("B must be a basis")
        elif not Vector.basis(*B_prime):
            raise BasisVectorsError("B_prime must be a basis")
        if len(B) != len(B_prime):
            raise BasisVectorsError("change_of_basis must have equal number of basis vectors")
        
        P = Matrix(*[vector.coordinates(*B_prime) for vector in B]).transpose()
        v = Matrix(self).transpose()

        return Vector(*((P @ v).transpose()))[0]
    
    def householder(self):
        I = Matrix.identity(self.dimension)
        if is_zero(self.norm):
            return I

        alpha = -sign(self.values[0]) * self.norm
        v = Matrix(self - alpha * I[0]).transpose()

        denom = (v.transpose() @ v)[0][0]

        if is_zero(denom):
            return I

        H = I - 2 * ((v @ v.transpose()) / denom )

        return H

        
    def __str__(self):
        return str(self.values)
    
    def __repr__(self):
        return f"Vector{self.values}"
    
    def __getitem__(self, key):
        return self.values[key]
    
    def __setitem__(self, key, value):
        self.values[key] = value
        return self
    
    def __delitem__(self, key):
        del self.values[key]
        return self

    def __neg__(self):
        return Vector(*(-a for a in self.values))

    def __pos__(self):
        return self
    
    def __len__(self):
        return len(self.values)
    
    def __round__(self, ndigits = 12):
        return Vector(*[round(entry, ndigits) for entry in self.values])
    
    def __eq__(self, other):
        return all(is_approx(a, b) for a, b in zip(self.values, other.values))
    
    def __add__(self, other):
        dimension_checker(self, other)
        return Vector(*(a + b for a, b in zip(self.values, other.values)))
    
    def __sub__(self, other):
        return self + -other
    
    def __mul__(self, other):
        if isinstance(other, (Vector, Matrix)):
            raise TypeError("__mul__ only accepts scalars")
        elif isinstance(other, (int, float)):
            return Vector(*(other * a for a in self.values))
        else:
            NotImplementedError(f"{type(other)} is not accounted for.")

    def __rmul__(self, other):
        return self * other
    
    def __matmul__(self, other):
        if isinstance(other, Vector):
            dimension_checker(self, other)
            return sum(a * b for a, b in zip(self.values, other.values))
        else:
            raise TypeError("__matmul__ accepts only vectors.")
        
    def __rmatmul__(self, other):
        raise TypeError("Dot product without a vector is not meaningful")
    
    def __truediv__(self, other):
        if isinstance(other, (Vector, Matrix)):
            raise TypeError("__truediv__ only accepts scalars")
        elif isinstance(other, (int, float)):
            return Vector(*(a / other for a in self.values))
        else:
            NotImplementedError(f"{type(other)} is not accounted for.")

    def __rtruediv__(self, other):
        raise TypeError("Division by Vector is not meaningful")
    
    @staticmethod
    def zero_vector(n):
        return Vector(*[0 for _ in range(n)])
    
    @staticmethod
    def linear_independence(*args):
        dimension_checker(*args)
        A = Matrix(*args)

        if A.columns < A.rows:
            return False
        
        return A.rank() == A.rows
    
    @staticmethod
    def span(*args):
        dimension_checker(*args)
        A = Matrix(*args)

        if A.columns > A.rows:
            return False
        
        return A.rank() == A.columns

    @staticmethod
    def basis(*args):
        return Vector.linear_independence(*args) and Vector.span(*args)

    @staticmethod
    def gram_schmidt(*args):

        A = [*args]
        B = [A[0]]

        for i in range(len(A)):
            for k in range(0, i):
                A[i] -= A[i].proj(B[k - 1])
                B.append(A[i])
        
        return A
    
    @staticmethod
    def gram_schmidt_orthonormal(*args):
        A = Vector.gram_schmidt(*args)
        B = []

        for i in range(len(A)):
            if is_zero(A[i].norm):
                continue
            B.append(A[i].normalize())

        return B
        

class Matrix(Tensor):


    def __init__(self, *args):
        if all(isinstance(arg, (list, tuple)) for arg in args):
            vectors = [Vector(*arg) for arg in args]
            vector_checker(*vectors)
            self.rowspace = vectors

        else:
            vector_checker(*args)
            self.rowspace = [arg for arg in args]
                

    @property
    def rows(self):
        return len(self.rowspace)   
    
    @property 
    def columns(self):
        return len(self.rowspace[0])
    
    def rank(self):
        return sum(1 for row in self.gauss_jordan() if any(val != 0 for val in row))
    
    def nullity(self):
        return self.columns - self.rank()
    
    def is_square(self):
        return self.rows == self.columns
    
    def is_triangular(self):
        return self.is_uppertriangular() or self.is_lowertriangular()

    def is_uppertriangular(self):
        return self.is_square() and all(is_zero(self[i][j])
                   for i in range(self.rows) 
                   for j in range(self.rows) 
                   if i > j)
    
    def is_lowertriangular(self):
        return self.is_square() and all(is_zero(self[i][j])
                   for i in range(self.rows) 
                   for j in range(self.rows) 
                   if i < j)
    
    def stabilize(self):
        A = copy.deepcopy(self)
        for i in range(A.rows):
            for j in range(A.columns):
                if is_zero(A[i][j]):
                    A[i][j] = 0
        return A
    
    def transpose(self):
        return Matrix(*[Vector(*[self[i][j] 
                for i in range(0, self.rows)]) 
                for j in range(0, self.columns)])   
    
    def gaussian(self, eps=EPS): # Added eps parameter
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
                if abs(factor) < eps:
                    A[i][j] = 0.0
                    continue
                A[i] = A[i] - factor * A[j]

        A = A.stabilize()
        return A
    
    def gauss_jordan(self, eps=EPS): # Added eps parameter
        A = copy.deepcopy(self).gaussian(eps=eps) # Pass eps here
        
        for j in range(min(A.rows, A.columns)):
            for i in range(A.rows):
                # Use is_nonapprox correctly (i is row, j is col)
                if i != j and abs(A[i][j]) > eps: 
                    A[i] = A[i] - A[i][j] * A[j]

        A = A.stabilize()
        return A

    def partition(self, other):
        if isinstance(other, Matrix):
            if not self.rows == other.rows:
                raise DimensionError("partition() only accepts matrices with equal rows")
            return Matrix(*[(a.partition(b)) 
                            for a, b in zip(self.rowspace, other.rowspace)])
        else:
            raise TypeError("Matrix can only be partitioned by Matrix class")
        
    def homogeneous(self, eps=1e-4): # Use the looser default
        A = copy.deepcopy(self)
        # Pass the looser eps to gauss_jordan
        A = A.gauss_jordan(eps=eps) 
        m, n = A.rows, A.columns

        # Zero out the matrix based on eps
        for i in range(m):
            for j in range(n):
                if is_zero(A[i][j], eps):
                    A[i][j] = 0.0
            
        pivot_map = [-1 for _ in range(n)]
        for i in range(m):
            for j in range(n):
                if is_nonzero(A[i][j], eps):
                    # Check if all previous in row are zero
                    if all(is_zero(A[i][k], eps) for k in range(j)):
                        pivot_map[j] = i
                        break

        basis_vectors = []

        for j in range(n):
            if pivot_map[j] == -1: # It is a free variable
                v = Vector(*[0 for _ in range(n)])
                v[j] = 1.0 
                
                for k in range(n):
                    # FIX: Compare index k to -1, not using is_nonapprox
                    if pivot_map[k] != -1: 
                        r = pivot_map[k]
                        # Correct sign: Ax = 0 => x_pivot = - A_pivot_free * x_free
                        v[k] = -A[r][j] 
                
                basis_vectors.append(v)

        return basis_vectors

    def qr(self):
        A = copy.deepcopy(self)
        Q = Matrix(*Vector.gram_schmidt_orthonormal(*A.transpose().rowspace)).transpose()
        R = Q.transpose() @ A

        Q, R = Q.stabilize(), R.stabilize()
        return Q, R
    
    def qr_householder(self):
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

        for i in range(0, A.rows):
            for j in range(0, A.columns): 
                A[i][j] += 0.0

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
        n = self.rows
        if self.gauss_jordan()[n - 1] == Vector.zero_vector(n):
            raise UninvertibleMatrixError()
        
        A = self.partition(Matrix.identity(n)).gauss_jordan()
        return Matrix(*(Vector(*(A[i][j] 
                                for j in range(n, 2*n))) 
                                for i in range(n)))

    @must_be_square
    def trace(self):
        return sum(self[i][i] for i in range(0, self.rows))

    @must_be_square
    def determinant(self):
        P, L, U = self.plu()
        return self._det_sign * math.prod(U[i][i] for i in range(self.rows))
    
    @must_be_square
    def eigenvalues(self):
        A = copy.deepcopy(self)
        n = A.rows
        I = Matrix.identity(n)

        if A.is_triangular():
            return [A[i][i] for i in range(n)]
        
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

        return [A[i][i] for i in range(n)]

    def eigenvectors(self):
        A = copy.deepcopy(self)
        eigenpairs = A.eigenpairs()

        eigenvectors = [v for value in eigenpairs.values() 
                        for v in (value if isinstance(value, list) 
                                  else [value])]

        return eigenvectors

    
    def eigenpairs(self):
        A = copy.deepcopy(self)
        I = Matrix.identity(A.rows)
        eigenvalues = A.eigenvalues()

        eigenpairs = {}
        for eigenvalue in eigenvalues:
            B = A - (eigenvalue * I)
            eigenpairs[eigenvalue] = B.homogeneous(eps=1e-4)

        return eigenpairs


    def diagonalize(self):
        A = copy.deepcopy(self)
        I = Matrix.identity(A.columns)
        eigenvalues = A.eigenvalues()
        eigenvectors = A.eigenvectors()

        if Matrix(*eigenvectors).rank() != A.columns:
            raise UndiagonalizableError("Matrix doesn't have " \
            "n linear independent eigenvectors")
        
        P = Matrix(*eigenvectors).transpose()
        D = I

        for i in range(A.columns):
            D[i][i] = eigenvalues[i]

        P, D = P.stabilize(), D.stabilize()
        return P, D, P.inverse()


    def __eq__(self, other):
        if isinstance(other, Matrix):
            return all(is_approx(a, b) for a, b in zip(self.rowspace, other.rowspace))
        else:
            return self == other
        
    def __round__(self, ndigits = 12):
        return Matrix(*[round(row, ndigits) for row in self.rowspace])
    
    def __str__(self):
        return '\n'.join(str(row) for row in self.rowspace)
    
    def __getitem__(self, key):
        return self.rowspace[key]
    
    def __setitem__(self, key, value):
        self.rowspace[key] = value
        return self
    
    def __delitem__(self, key):
        del self.rowspace[key]
        return self
    
    def __neg__(self):
        return Matrix(*(-a for a in self.rowspace))
    
    def __pos__(self):
        return self
    
    def __add__(self, other):
        validate_matrix_compatibility(self, other)
        return Matrix(*(a + b for a, b in zip(self.rowspace, other.rowspace)))
    
    def __sub__(self, other):
        return self + -other
    
    def __mul__(self, other):
        if isinstance(other, (Matrix, Vector)):
            return TypeError("__mul__ only accepts scalars.")
        elif isinstance(other, (int, float)):
            return Matrix(*(other * a for a in self.rowspace))
        else:
            NotImplementedError(f"{type(other)} is not accounted for.")

    def __matmul__(self, other):
        if isinstance(other, Matrix):
            validate_matrix_mult_compatibility(self, other)
            return round(Matrix(*[Vector(*[self[j] @ other.transpose()[i] 
                 for j in range(0, self.rows)]) 
                 for i in range(0, other.columns)]).transpose().stabilize())
        else:
            raise TypeError("__matmul__ accepts only matrices.")
        
    def __rmatmul__(self, other):
        raise TypeError("Matrix Multiplication without a matrix is not meaningful")
    
    def __pow__(self, other):
        if isinstance(other, (int)):
            result = self
            for _ in range(other - 1):
                result = result @ self
            return result
        else:
            raise TypeError("Matrix exponentiation is only meaningful" \
            "with integer powers.")

    def __rmul__(self, other):
        return self * other
    
    def __truediv__(self, other):
        if isinstance(other, (Vector, Matrix)):
            raise TypeError("__truediv__ only accepts scalars")
        elif isinstance(other, (int, float)):
            return Matrix(*(a / other for a in self.rowspace))
        else:
            NotImplementedError(f"{type(other)} is not accounted for.")

    def __rtruediv__(self, other):
        raise TypeError("Division by Matrix is not meaningful")
    
    @staticmethod
    def solve(A, b):
        if not isinstance(A, Matrix):
            raise TypeError("A must be a Matrix")
        elif not isinstance(b, Vector):
            raise TypeError("b must be a Vector")
        
        C = A.partition(Matrix(b).transpose()).gauss_jordan()

        for row in C:
            if all(is_zero(x) for x in row[:-1]) and is_nonzero(row[-1]):
                raise InconsistentSystemError()
        
        if A.rank() < A.columns:
            raise InfiniteSolutionsError()
        
        x = Vector(*[C[i][C.columns - 1] for i in range(C.rows)])
        return x
            
    @staticmethod
    def identity(n):
        return Matrix(*[Vector(*[1 if i == j else 0 
                                 for j in range(n)]) 
                                 for i in range(n)])
   


