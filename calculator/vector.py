
import math
from .exceptions import BasisVectorsError
from .utils import *

class Vector:
    """A class representing a vector in linear algebra.

    Attributes:
        values (list): The components of the vector.
    """

    def __init__(self, *args):
        """Initializes a Vector.

        Args:
            *args: Components of the vector, either as separate args or a single iterable.
        """
        if isinstance(args, (tuple, list)):
            self.values = list(args)
        else:
            self.values = []
            entry_checker(*args)
            for arg in args:
                self.values.append(arg)

    @property
    def norm(self):
        """The Euclidean norm of the vector.

        Returns:
            float: The norm.
        """
        return self.dot(self) ** 0.5
    
    @property
    def dimension(self):
        """The dimension of the vector.

        Returns:
            int: The dimension.
        """
        return len(self.values)
    
    def normalize(self):
        """Normalizes the vector.

        Returns:
            Vector: The normalized vector.

        Raises:
            ValueError: If the vector has zero norm.
        """
        if self.norm == 0:
            raise ValueError("Cannot normalize vector with 0 norm")
        return Vector(*(a / self.norm for a in self.values))
    
    def dot(self, other):
        """Computes the dot product with another vector.

        Args:
            other (Vector): The other vector.

        Returns:
            float: The dot product.
        """
        dimension_checker(self, other)
        return round(sum((a * b for a, b in zip(self.values, other.values))), 10)
    
    def angle(self, other):
        """Computes the angle with another vector.

        Args:
            other (Vector): The other vector.

        Returns:
            float: The angle in radians.
        """
        dimension_checker(self, other)
        return math.acos((self.dot(other)) / (self.norm * other.norm))
    
    def proj(self, other):
        """Projects this vector onto another vector.

        Args:
            other (Vector): The vector to project onto.

        Returns:
            Vector: The projection.
        """
        dimension_checker(self, other)
        if is_zero((other.norm ** 2)):
            return Vector.zero_vector(self.dimension)
        return (self.dot(other)/(pow(other.norm, 2))) * other
    
    def orthogonal_decomp(self, other):
        """Performs orthogonal decomposition.

        Args:
            other (Vector): The vector to decompose against.

        Returns:
            tuple: (projection, orthogonal component)
        """
        return self.proj(other), self - self.proj(other)
    
    def partition(self, other):
        """Partitions this vector with another.

        Args:
            other (Vector): The other vector.

        Returns:
            Vector: The concatenated vector.

        Raises:
            TypeError: If other is not a Vector.
        """
        if isinstance(other, Vector):
            return Vector(*[*self.values, *other.values])
        else:
            raise TypeError("Vector can only be partitioned by Vector class")
    
    def coordinates(self, *args):

        from .matrix import Matrix

        """Computes coordinates in the given basis.

        Args:
            *args: Basis vectors.

        Returns:
            Vector: The coordinates.

        Raises:
            BasisVectorsError: If args do not form a basis.
        """
        if not Vector.basis(*args): raise BasisVectorsError("*args must form a basis")

        A = Matrix(*args).transpose()
        return Matrix.solve(A, self)
    
    def change_of_basis(self, B, B_prime):

        from .matrix import Matrix

        """Changes the basis of the vector.

        Args:
            B: Original basis.
            B_prime: New basis.

        Returns:
            Vector: The vector in the new basis.

        Raises:
            BasisVectorsError: If bases are invalid.
        """
        if not Vector.basis(*B):
            raise BasisVectorsError("B must be a basis")
        elif not Vector.basis(*B_prime):
            raise BasisVectorsError("B_prime must be a basis")
        if len(B) != len(B_prime):
            raise BasisVectorsError("Bases inputted to change_of_basis " \
            "must have equal number of basis vectors")
        
        P = Matrix(*[vector.coordinates(*B_prime) for vector in B]).transpose()
        v = Matrix(self).transpose()

        return Vector(*((P @ v).transpose()))[0]
    
    def householder(self):

        from .matrix import Matrix

        """Computes the Householder reflection matrix.

        Returns:
            Matrix: The Householder matrix.
        """
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
        """String representation of the vector.

        Returns:
            str: The string.
        """
        return str(self.values)
    
    def __repr__(self):
        """Representation of the vector.

        Returns:
            str: The representation.
        """
        return f"Vector{self.values}"
    
    def __getitem__(self, key):
        """Gets an item by index.

        Args:
            key: The index.

        Returns:
            The item.
        """
        return self.values[key]
    
    def __setitem__(self, key, value):
        """Sets an item by index.

        Args:
            key: The index.
            value: The value.

        Returns:
            Vector: Self.
        """
        self.values[key] = value
        return self
    def __delitem__(self, key):
        """Deletes an item by index.

        Args:
            key: The index.

        Returns:
            Vector: Self.
        """
        del self.values[key]
        return self
    def __neg__(self):
        """Negates the vector.

        Returns:
            Vector: The negated vector.
        """
        return Vector(*(-a for a in self.values))

    def __pos__(self):
        """Positive operator.

        Returns:
            Vector: Self.
        """
        return self

    def __len__(self):
        """Length of the vector.

        Returns:
            int: The length.
        """
        return len(self.values)
    
    def __round__(self, ndigits = 12):
        """Rounds the vector components.

        Args:
            ndigits (int): Number of digits.

        Returns:
            Vector: The rounded vector.
        """
        return Vector(*[round(entry, ndigits) for entry in self.values])
    def __eq__(self, other):
        """Checks equality with another vector.

        Args:
            other: The other vector.

        Returns:
            bool: True if equal.
        """
        return all(is_approx(a, b) for a, b in zip(self.values, other.values))
    def __add__(self, other):
        """Adds two vectors.

        Args:
            other (Vector): The other vector.

        Returns:
            Vector: The sum.
        """
        dimension_checker(self, other)
        return Vector(*(a + b for a, b in zip(self.values, other.values)))
    def __sub__(self, other):
        """Subtracts another vector.

        Args:
            other (Vector): The other vector.

        Returns:
            Vector: The difference.
        """
        return self + -other

    def __mul__(self, other):

        from .matrix import Matrix

        """Multiplies by a scalar.

        Args:
            other: The scalar.

        Returns:
            Vector: The scaled vector.

        Raises:
            TypeError: If other is not a scalar.
            NotImplementedError: For unsupported types.
        """
        if isinstance(other, (Vector, Matrix)):
            raise TypeError("__mul__ only accepts scalars")
        elif isinstance(other, (int, float)):
            return Vector(*(other * a for a in self.values))
        else:
            raise NotImplementedError(f"{type(other)} is not accounted for.")

    def __rmul__(self, other):
        """Right multiplication by scalar.

        Args:
            other: The scalar.

        Returns:
            Vector: The scaled vector.
        """
        return self * other

    def __matmul__(self, other):
        """Matrix multiplication with vector (dot product).

        Args:
            other (Vector): The other vector.

        Returns:
            float: The dot product.

        Raises:
            TypeError: If other is not a Vector.
        """
        if isinstance(other, Vector):
            dimension_checker(self, other)
            return sum(a * b for a, b in zip(self.values, other.values))
        else:
            raise TypeError("__matmul__ accepts only vectors.")
        
    def __rmatmul__(self, other):
        """Right matrix multiplication.

        Raises:
            TypeError: Always.
        """
        raise TypeError("Dot product without a vector is not meaningful")
    
    def __truediv__(self, other):

        from .matrix import Matrix

        """Divides by a scalar.

        Args:
            other: The scalar.

        Returns:
            Vector: The divided vector.

        Raises:
            TypeError: If other is not a scalar.
            NotImplementedError: For unsupported types.
        """
        if isinstance(other, (Vector, Matrix)):
            raise TypeError("__truediv__ only accepts scalars")
        elif isinstance(other, (int, float)):
            return Vector(*(a / other for a in self.values))
        else:
            NotImplementedError(f"{type(other)} is not accounted for.")
    def __rtruediv__(self, other):
        """Right division.

        Raises:
            TypeError: Always.
        """
        raise TypeError("Division by Vector is not meaningful")
    @staticmethod
    def zero_vector(n):
        """Creates a zero vector.

        Args:
            n (int): The dimension.

        Returns:
            Vector: The zero vector.
        """
        return Vector(*[0 for _ in range(n)])

    @staticmethod
    def linear_independence(*args):

        from .matrix import Matrix

        """Checks if vectors are linearly independent.

        Args:
            *args: The vectors.

        Returns:
            bool: True if linearly independent.
        """
        dimension_checker(*args)
        A = Matrix(*args)

        if A.columns < A.rows:
            return False
        
        return A.rank() == A.rows
    
    @staticmethod
    def span(*args):

        from .matrix import Matrix

        """Checks if vectors span the space.

        Args:
            *args: The vectors.

        Returns:
            bool: True if they span.
        """
        dimension_checker(*args)
        A = Matrix(*args)

        if A.columns > A.rows:
            return False
        
        return A.rank() == A.columns

    @staticmethod
    def basis(*args):
        """Checks if vectors form a basis.

        Args:
            *args: The vectors.

        Returns:
            bool: True if they form a basis.
        """
        return Vector.linear_independence(*args) and Vector.span(*args)
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
        
        """Performs orthonormal Gram-Schmidt.

        Args:
            *args: The vectors.

        Returns:
            list: The orthonormal vectors.
        """
        A = Vector.gram_schmidt(*args)
        B = []

        for i in range(len(A)):
            if is_zero(A[i].norm):
                continue
            B.append(A[i].normalize())

        return B