
import unittest

from calculator import *

def matrix_to_values(matrix):
    return [[matrix[i][j] for j in range(matrix.columns)] for i in range(matrix.rows)]


class TestVectorOperations(unittest.TestCase):
    def assertVectorAlmostEqual(self, actual, expected, eps=1e-8):
        self.assertEqual(len(actual), len(expected))
        for a, b in zip(actual, expected):
            self.assertTrue(is_approx(a, b, eps), f"{a} != {b}")

    def test_vector_arithmetic_norm_and_scalar_ops(self):
        u = Vector(1, 2)
        v = Vector(3, 4)

        self.assertEqual(u + v, Vector(4, 6))
        self.assertEqual(v - u, Vector(2, 2))
        self.assertEqual(2 * u, Vector(2, 4))
        self.assertEqual(v / 2, Vector(1.5, 2.0))
        self.assertEqual(u.dot(v), 11)
        self.assertEqual(u @ v, 11)
        self.assertTrue(is_approx(u.norm, 2.23606797749979))
        self.assertEqual(u.dimension, 2)

    def test_vector_normalize_and_angle(self):
        u = Vector(3, 4)
        normalized = u.normalize()

        self.assertVectorAlmostEqual(normalized, Vector(0.6, 0.8))
        self.assertTrue(is_approx(normalized.norm, 1.0))
        self.assertTrue(is_approx(u.angle(Vector(3, 0)), 0.9272952180016123))

    def test_vector_projection_and_orthogonal_decomposition(self):
        u = Vector(3, 4)
        v = Vector(1, 0)

        projection = u.proj(v)
        self.assertEqual(projection, Vector(3.0, 0.0))

        proj_component, orth_component = u.orthogonal_decomp(v)
        self.assertEqual(proj_component, Vector(3.0, 0.0))
        self.assertEqual(orth_component, Vector(0.0, 4.0))

    def test_vector_basis_coordinates_and_change_of_basis(self):
        v = Vector(3, 4)
        basis = [Vector(1, 0), 
                 Vector(0, 1)]

        self.assertTrue(Vector.basis(*basis))
        self.assertEqual(v.coordinates(*basis), Vector(3.0, 4.0))
        self.assertTrue(Vector.linear_independence(*basis))
        self.assertTrue(Vector.span(*basis))

        rotated_basis = [Vector(1, 1), Vector(-1, 1)]
        self.assertTrue(Vector.basis(*rotated_basis))

    def test_vector_zero_normalize_raises_error(self):
        with self.assertRaises(ValueError):
            Vector.zero_vector(3).normalize()

    def test_vector_partition_rejects_invalid_type(self):
        with self.assertRaises(TypeError):
            Vector(1, 2).partition(5)


class TestMatrixOperations(unittest.TestCase):
    def assertMatrixAlmostEqual(self, actual, expected, eps=1e-8):
        self.assertEqual(actual.rows, expected.rows)
        self.assertEqual(actual.columns, expected.columns)
        for i in range(actual.rows):
            for j in range(actual.columns):
                self.assertTrue(
                    is_approx(actual[i][j], expected[i][j], eps),
                    f"Matrix entries differ at {i},{j}: {actual[i][j]} != {expected[i][j]}",
                )

    def test_matrix_construction_properties_and_transpose(self):
        A = Matrix([1, 2], 
                   [3, 4])
        
        B = Matrix([1, 0], 
                   [0, 1])
        
        self.assertEqual(A.rows, 2)
        self.assertEqual(A.columns, 2)
        self.assertTrue(A.is_square())
        self.assertFalse(A.is_triangular())
        self.assertTrue(B.is_symmetric())

        AT = A.transpose()
        self.assertMatrixAlmostEqual(AT, Matrix([1, 3], 
                                                [2, 4]))

    def test_matrix_gaussian_rank_nullity_and_rref(self):
        A = Matrix([1, 2, 3], 
                   [2, 4, 6])

        self.assertEqual(A.rank(), 1)
        self.assertEqual(A.nullity(), 2)

        ref = A.gaussian()
        self.assertTrue(ref[0][0] == 1.0)
        self.assertTrue(is_approx(ref[1][0], 0.0))

        rref = A.gauss_jordan()
        self.assertMatrixAlmostEqual(rref, Matrix([1.0, 2.0, 3.0], 
                                                  [0.0, 0.0, 0.0]))

    def test_matrix_partition_and_homogeneous(self):
        A = Matrix([1, 2, 3], 
                   [2, 4, 6])
        H = A.homogeneous()

        expected = [Vector(-2.0, 1.0, 0), Vector(-3.0, 0, 1.0)]
        self.assertEqual(H, expected)

        B = Matrix([1, 0], 
                   [0, 1])
        I = Matrix.identity(2)
        self.assertMatrixAlmostEqual(B.partition(I), Matrix([1, 0, 1, 0], 
                                                            [0, 1, 0, 1]))

    def test_matrix_solve_unique_inconsistent_and_zero_rhs(self):
        A = Matrix(Vector(1, 2), 
                   Vector(3, 4))
        b = Vector(5, 11)
        solution = Matrix.solve(A, b)
        self.assertEqual(solution, Vector(1.0, 2.0))

        inconsistent = Matrix(Vector(1, 2), Vector(2, 4))
        with self.assertRaises(InconsistentSystemError):
            Matrix.solve(inconsistent, Vector(1, 0))

        self.assertEqual(Matrix.solve(A, Vector.zero_vector(2)), A.homogeneous())

    def test_matrix_plu_inverse_trace_and_determinant(self):
        A = Matrix(Vector(0, 2), Vector(1, 3))
        P, L, U = A.plu()

        self.assertMatrixAlmostEqual(P @ L @ U, A)
        self.assertTrue(L.is_lowertriangular())
        self.assertTrue(U.is_uppertriangular())
        self.assertEqual(A.trace(), 3)
        self.assertTrue(is_approx(A.determinant(), -2.0))

        invA = Matrix(Vector(-1.5, 1.0), Vector(0.5, 0.0))
        self.assertMatrixAlmostEqual(A.inverse(), invA)

        singular = Matrix(Vector(1, 2), Vector(2, 4))
        with self.assertRaises(UninvertibleMatrixError):
            singular.inverse()

    def test_matrix_qr_and_eigen_decomposition(self):
        A = Matrix(Vector(1, 2), Vector(3, 4))
        Q, R = A.qr()

        self.assertTrue(Q.is_square())
        self.assertTrue(R.is_uppertriangular())
        self.assertMatrixAlmostEqual(Q @ R, A)

        diag = Matrix(Vector(2, 0), Vector(0, 3))
        self.assertEqual(diag.eigenvalues(), [3, 2])

    def test_matrix_diagonalization_and_orthogonal_diagonalization(self):
        A = Matrix(Vector(2, 0), Vector(0, 3))
        P, D, Pinv = A.diagonalization()

        self.assertMatrixAlmostEqual(P, Matrix(Vector(0, 1.0), Vector(1.0, 0)))
        self.assertMatrixAlmostEqual(D, Matrix(Vector(3, 0), Vector(0, 2)))
        self.assertMatrixAlmostEqual(Pinv, P)
        self.assertMatrixAlmostEqual(P @ D @ Pinv, A)

        Q, Dq, Qinv = A.orthogonal_diagonalization()
        self.assertMatrixAlmostEqual(Q @ Dq @ Qinv, A)
        self.assertMatrixAlmostEqual(Qinv, Q)

    def test_matrix_symmetric_diagonalization_error(self):
        A = Matrix(Vector(1, 2), Vector(3, 4))
        with self.assertRaises(UndiagonalizableError):
            A.orthogonal_diagonalization()

    def test_matrix_error_for_invalid_partition_type(self):
        with self.assertRaises(TypeError):
            Matrix(Vector(1, 2), Vector(3, 4)).partition(5)


if __name__ == '__main__':
    unittest.main()
