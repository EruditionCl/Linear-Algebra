
from .exceptions import DimensionError

def dimension_checker(*args):
    """Checks if all input vectors have the same dimension.

    Args:
        *args: Variable number of vectors to check.

    Raises:
        DimensionError: If dimensions do not match.
    """
    dimensions = [arg.dimension for arg in args]
    if len(set(dimensions)) != 1:
        raise DimensionError("Mismatching Dimensions")

def validate_matrix_compatibility(A, B):
    """Validates that two matrices have the same dimensions.

    Args:
        A (Matrix): First matrix.
        B (Matrix): Second matrix.

    Raises:
        DimensionError: If dimensions do not match.
    """
    if not (A.rows, A.columns) == (B.rows, B.columns):
        raise DimensionError("Mismatching Dimensions")

def validate_matrix_mult_compatibility(A, B):
    """Validates compatibility for matrix multiplication.

    Args:
        A (Matrix): First matrix.
        B (Matrix): Second matrix.

    Raises:
        DimensionError: If A.columns != B.rows.
    """
    if not (A.columns) == (B.rows):
        raise DimensionError("Mismatching Dimensions")

def vector_checker(*args):

    from .vector import Vector

    """Checks if all arguments are Vector instances and have same dimension.

    Args:
        *args: Variable number of arguments to check.

    Raises:
        TypeError: If any argument is not a Vector.
        DimensionError: If dimensions do not match.
    """
    if args == None:
        return
    for arg in args:
        if not isinstance(arg, Vector):
            raise TypeError("Only Vector entries accepted")
        dimension_checker(*args)
    
def entry_checker(*args):
    """Checks if all arguments are int or float.

    Args:
        *args: Variable number of arguments to check.

    Raises:
        TypeError: If any argument is not int or float.
    """
    if not all(isinstance(arg, (int,float)) for arg in args):
        raise TypeError("Vector class only accepts int and float")
    

def sign(n):
    """Returns the sign of a number.

    Args:
        n: A number.

    Returns:
        int: 1 if n >= 0, -1 otherwise.
    """
    return 1 if n >= 0 else -1

def distance(u, v):
    """Computes the Euclidean distance between two vectors.

    Args:
        u (Vector): First vector.
        v (Vector): Second vector.

    Returns:
        float: The distance.
    """
    vector_checker(u, v)
    return (u-v).norm
    
def dot(u, v):
    """Computes the dot product of two vectors.

    Args:
        u (Vector): First vector.
        v (Vector): Second vector.

    Returns:
        float: The dot product.
    """
    return u.dot(v)

def angle(u, v):
    """Computes the angle between two vectors.

    Args:
        u (Vector): First vector.
        v (Vector): Second vector.

    Returns:
        float: The angle in radians.
    """
    return u.angle(v)

def proj(u, v):
    """Computes the projection of vector u onto vector v.

    Args:
        u (Vector): Vector to project.
        v (Vector): Vector to project onto.

    Returns:
        Vector: The projection vector.
    """
    return u.proj(v)

def partition(u, v):
    """Partitions two vectors into one.

    Args:
        u (Vector): First vector.
        v (Vector): Second vector.

    Returns:
        Vector: Concatenated vector.
    """
    return u.partition(v)

EPS = 1e-10

def is_nonapprox(a, b, eps=EPS):
    """Checks if two numbers are not approximately equal.

    Args:
        a: First number.
        b: Second number.
        eps (float): Tolerance.

    Returns:
        bool: True if not approximately equal.
    """
    return abs(a - b) >= eps * max(1.0, abs(a), abs(b))

def is_approx(a, b, eps=EPS):
    """Checks if two numbers are approximately equal.

    Args:
        a: First number.
        b: Second number.
        eps (float): Tolerance.

    Returns:
        bool: True if approximately equal.
    """
    return abs(a - b) < eps * max(1.0, abs(a), abs(b))

def is_zero(x, eps = EPS):
    """Checks if a number is approximately zero.

    Args:
        x: The number.
        eps (float): Tolerance.

    Returns:
        bool: True if approximately zero.
    """
    return abs(x) < eps

def is_nonzero(x, eps = EPS):
    """Checks if a number is not approximately zero.

    Args:
        x: The number.
        eps (float): Tolerance.

    Returns:
        bool: True if not approximately zero.
    """
    return abs(x) >= eps

def must_be_square(func):
    """Decorator to ensure the matrix is square.

    Args:
        func: The function to decorate.

    Returns:
        function: The decorated function.
    """
    def inner(self, *args, **kwargs):
        if not self.is_square():
            raise DimensionError("Matrix must be square")
        return func(self, *args, **kwargs)
    return inner