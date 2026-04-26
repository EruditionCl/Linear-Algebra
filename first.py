
import math

def DimensionChecker(*args):
    dimensions = []
    for arg in args:
        dimensions.append(arg.dimension)
    if len(set(dimensions)) != 1:
        raise DimensionError("Mismatching Dimensions")

def ValidateMatrixCompatibility(A, B):
    if not (A.rows, A.columns) == (B.rows, B.columns):
        raise DimensionError("Mismatching Dimensions")

def ValidateMatrixMultCompatibility(A, B):
    if not (A.rows) == (B.columns):
        raise DimensionError("Mismatching Dimensions")

def VectorChecker(*args):
    if args == None:
        return
    for arg in args:
        if not isinstance(arg, Vector):
            raise TypeError("Only Vector entries accepted")
        DimensionChecker(*args)
    
def EntryChecker(*args):
    if not all(isinstance(arg, (int,float)) for arg in args):
        raise TypeError("Vector class only accepts int and float")
    



def distance(u, v):
    VectorChecker(u, v)
    return (u-v).norm
    
def dot(u, v):
    return u.dot(v)

def angle(u, v):
    return u.angle(v)

def proj(u, v):
    return u.proj(v)


    
class DimensionError(Exception):
    pass

class Vector:
    def __init__(self, *args):
        if isinstance(args, (tuple, list)):
            self.values = list(args)
        else:
            self.values = []
            EntryChecker(*args)
            for arg in args:
                self.values.append(arg)
        self.dimension = len(self.values)

    @property
    def norm(self):
        return self.dot(self) ** 0.5
    
    def normalize(self):
        return Vector(*(a / self.norm for a in self.values))
    
    def dot(self, other):
        DimensionChecker(self, other)
        return sum((a * b for a, b in zip(self.values, other.values)))
    
    def angle(self, other):
        DimensionChecker(self, other)
        return math.acos(round((self.dot(other)) / (self.norm * other.norm)))
    
    def proj(self, other):
        DimensionChecker(self, other)
        return (self.dot(other)/(pow(other.norm, 2))) * other
    
    def orthogonaldecomp(self, other):
        return (self.proj(other), self - self.proj(other))

    def __str__(self):
        return str(self.values)
    
    def __repr__(self):
        return f"Vector{self.values}"
    
    def __getitem__(self, key):
        return self.values[key-1]
    
    def __setitem__(self, key, value):
        self.values[key-1] = value
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
        DimensionChecker(self, other)
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
            DimensionChecker(self, other)
            return Vector(*(a * b for a, b in zip(self.values, other.values)))
        else:
            raise TypeError("__matmul__ accepts only vectors.")
    
    def __truediv__(self, other):
        if isinstance(other, (Vector, Matrix)):
            raise TypeError("__truediv__ only accepts scalars")
        elif isinstance(other, (int, float)):
            return Vector(*(a/other for a in self.values))
        else:
            NotImplementedError(f"{type(other)} is not accounted for.")

    def __rtruediv__(self, other):
        raise TypeError("Division by Vector is not meaningful")


class Matrix:

    def __new__(cls, *args):
        instance = super().__new__(cls)
        instance.__init__(*args)

        if all(isinstance(arg, (list, tuple)) for arg in args):
            vectors = [Vector(*arg) for arg in args]
            VectorChecker(*vectors)
            instance.rowspace = vectors

        else:
            VectorChecker(*args)
            instance.rowspace = [arg for arg in args]

        instance.rows = len(instance.rowspace)
        instance.columns = len(instance.rowspace[0])

        if instance.rows == instance.columns:
            instance.__class__ = SquareMatrix
        return instance   

    def __init__(self, *args):
        pass
    
    def transpose(self):
        return Matrix(*[Vector(*[self[i][j] for i in range(1, self.rows + 1)]) 
                for j in range(1, self.columns + 1)])   
    
    def gaussian(self):
        for j in range(1, self.rows + 1):
            if self[j][j] == 0:
                self[j], self[j-1] = self[j-1], self[j]
            self[j] = self[j] / self[j][j]
            for i in range(j + 1, self.rows + 1):
                self[i] = self[i] - self[i][j] * self[j]

        for i in range(1, self.rows + 1):
            for j in range(1, self.rows + 1):
                self[i][j] += 0.0
        return round(self)
    
    def gaussjordan(self):
        self.gaussian()
        for i in range(self.rows, 1, -1):
            for j in range(1, i):
                self[j] = self[j] - self[j][i]*self[i]
        for i in range(1, self.rows + 1):
            for j in range(1, self.rows + 1):
                self[i][j] += 0.0
        return round(self)

    def __eq__(self, other):
        if isinstance(other, Matrix):
            return all(a == b for a, b in zip(self.rowspace, other.rowspace))
        else:
            return self == other
        
    def __round__(self, ndigits=10):
        return Matrix(*[round(row, ndigits) for row in self.rowspace])
    
    def __str__(self):
        return '\n'.join(str(row) for row in self.rowspace)
    
    def __getitem__(self, key):
        return self.rowspace[key-1]
    
    def __setitem__(self, key, value):
        self.rowspace[key-1] = value
        return self
    
    def __neg__(self):
        return Matrix(*(-a for a in self.rowspace))
    
    def __pos__(self):
        return self
    
    def __add__(self, other):
        ValidateMatrixCompatibility(self, other)
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
            ValidateMatrixMultCompatibility(self, other)
            return Matrix(*(a @ b for a, b in zip(self.rowspace, other.rowspace)))
        else:
            raise TypeError("__matmul__ accepts only matrices.")

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
   
class SquareMatrix(Matrix):
    def trace(self):
        return sum(self[i][i] for i in range(1, self.rows + 1))



A = Matrix ((1, 2, 3),
            (1, 2, 3),
            (1, 2, 3),
            (1, 2, 3))

B = Matrix(Vector(3, 0, 0, 0),
           Vector(0, 3, 0, 0),
           Vector(0, 0, 3, 0))




print(A @ B)






             
