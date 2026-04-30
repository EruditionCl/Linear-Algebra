
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

class Vector:
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
    
    def __round__(self, ndigits=10):
        return Vector(*[round(entry, ndigits) for entry in self.values])
    
    def __eq__(self, other):
        return all(a == b for a, b in zip(self.values, other.values))
    
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
            return Vector(*(a/other for a in self.values))
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
        if not Vector.basis(*args):
            raise BasisVectorsError("gram_schmidt only accepts basis vectors")

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
        return [A[i].normalize() for i in range(len(A))]
        

class Matrix:

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
        return self.is_square() and all(self[i][j] == 0 
                   for i in range(self.rows) 
                   for j in range(self.rows) 
                   if i > j)
    
    def is_lowertriangular(self):
        return self.is_square() and all(self[i][j] == 0 
                   for i in range(self.rows) 
                   for j in range(self.rows) 
                   if i < j)
    
    def transpose(self):
        return Matrix(*[Vector(*[self[i][j] 
                for i in range(0, self.rows)]) 
                for j in range(0, self.columns)])   
    
    def gaussian(self):
        A = copy.deepcopy(self)
        for j in range(0, min(A.rows, A.columns)):
            if A[j][j] == 0:
                for k in range(j + 1, A.rows):
                    if A[k][j] != 0:
                        A[j], A[k] = A[k], A[j]
                        break
            
            if A[j][j] != 0:
                A[j] = A[j] / A[j][j]
                
                for i in range(j + 1, A.rows):
                    A[i] = A[i] - A[i][j] * A[j]

        for i in range(0, A.rows):
            for j in range(0, A.columns): 
                A[i][j] += 0.0
        return round(A)
    
    def gauss_jordan(self):
        A = copy.deepcopy(self).gaussian()
        
        for j in range(min(A.rows, A.columns)):
                for i in range(A.rows):
                    if i != j:
                        A[i] = A[i] - A[i][j] * A[j]
        
        for i in range(A.rows):
            for j in range(A.columns):
                A[i][j] += 0.0
        return round(A)   

    def partition(self, other):
        if isinstance(other, Matrix):
            if not self.rows == other.rows:
                raise DimensionError("partition() only accepts matrices with equal rows")
            return Matrix(*[(a.partition(b)) 
                            for a, b in zip(self.rowspace, other.rowspace)])
        else:
            raise TypeError("Matrix can only be partitioned by Matrix class")
        
    @must_be_square
    def plu(self):
        n = self.rows
        P = Matrix.identity(n)
        A = copy.deepcopy(self)
        det_sign = 1

        for j in range(0, n):
            if A[j][j] == 0:
                for k in range(j + 1, n):
                    if A[k][j] != 0:
                        A[j], A[k] = A[k], A[j]
                        P[j], P[k] = P[k], P[j]
                        det_sign *= -1
                        break

            if A[j][j] != 0:
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
        return P, L, U
        
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
        
    def __eq__(self, other):
        if isinstance(other, Matrix):
            return all(a == b  for a, b in zip(self.rowspace, other.rowspace))
        else:
            return self == other
        
    def __round__(self, ndigits=10):
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
            return Matrix(*[Vector(*[self[j] @ other.transpose()[i] 
                 for j in range(0, self.rows)]) 
                 for i in range(0, other.columns)]).transpose()
        else:
            raise TypeError("__matmul__ accepts only matrices.")
        
    def __rmatmul__(self, other):
        raise TypeError("Matrix Multiplication without a matrix is not meaningful")
    
    def __pow__(self, other):
        if isinstance(other, (int)):
            result = self
            for _ in range(other-1):
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
            return Matrix(*(a/other for a in self.rowspace))
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
            if all(x == 0 for x in row[:-1]) and row[-1] != 0:
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
   

v1 = Vector(1, 0, 0, 0, 1)
v2 = Vector(0, 1, 0, 1, 0)
v3 = Vector(0, 0, 1, 1, 1)
v4 = Vector(1, 1, 0, 0, 0)
v5 = Vector(0, 1, 1, 0, 0)

v, w, x, y, z = Vector.gram_schmidt_orthonormal(v1, v2, v3, v4, v5)
vectors = [v,w,x,y,z]
print(vectors)
    













             
