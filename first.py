
import math

def DimensionChecker(*args):
    dimensions = []
    for arg in args:
        dimensions.append(arg.dimension)
    if len(set(dimensions)) != 1:
        raise DimensionError("Mismatching Dimensions")
    
def MatrixDimensionChecker(*args):
    dimensions = []
    for arg in args:
        dimensions.append(arg.dimension)
    if len(set(dimensions)) != 1:
        raise DimensionError("Mismatching Dimensions")

def VectorChecker(*args):
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

    def __neg__(self):
        return Vector(*(-a for a in self.values))

    def __pos__(self):
        return self
    
    def __len__(self):
        return len(self.values)
    
    def __eq__(self, other):
        return all(a == b for a, b in zip(self.values, other.values))
    
    def __add__(self, other):
        DimensionChecker(self, other)
        return Vector(*(a + b for a, b in zip(self.values, other.values)))
    
    def __sub__(self, other):
        return self + -other
    
    def __mul__(self, other):
        if isinstance(other, Vector):
            raise TypeError("__mul__ only accepts scalars")
        elif isinstance(other, (int, float)):
            return Vector(*(other * a for a in self.values))
        else:
            NotImplementedError(f"{type(other)} is not accounted for.")

    def __rmul__(self, other):
        return self * other
    
    def __truediv__(self, other):
        if isinstance(other, Vector):
            raise TypeError("__truediv__ only accepts scalars")
        elif isinstance(other, (int, float)):
            return Vector(*(a/other for a in self.values))
        else:
            NotImplementedError(f"{type(other)} is not accounted for.")

    def __rtruediv__(self, other):
        raise TypeError("A scalar can't be divided by Vector")


class Matrix:
    def __init__(self, *args):
        self.rowspace = []
        VectorChecker(*args)
        for arg in args:
            self.rowspace.append(arg)
        self.rows = len(self.rowspace)
        self.columnspace = []
        for i in range(self.rows):
            for j in range(len(self.rowspace[i])):
                self.columnspace.append(Vector(self.rowspace[j][i]))
        print(self.columnspace)
    
    def __str__(self):
        return '\n'.join(str(row) for row in self.rowspace)
    
    def __getitem__(self, key):
        return self.rowspace[key-1]
    
    def __add__(self, other):
        return Matrix(*(a + b for a, b in zip(self.rowspace, other.rowspace)))
    
    


A = Matrix(Vector(3, 4, 0),
           Vector(3, 3, 0),
           Vector(0, 4, 3),
           Vector(0, 2, 3))


print(A)

             
