
class DimensionError(Exception):
    """Exception raised for dimension mismatches."""
    pass

class UninvertibleMatrixError(Exception):
    """Exception raised when a matrix is not invertible."""
    pass

class InconsistentSystemError(Exception):
    """Exception raised for inconsistent linear systems."""
    pass

class BasisVectorsError(Exception):
    """Exception raised for basis vector errors."""
    pass

class UndiagonalizableError(Exception):
    """Exception raised when a matrix is not diagonalizable."""
    pass