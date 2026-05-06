
from first import *

A = Matrix([1, 2, 0],
           [0, 1, 1],
           [1, 0, 1],
           [2, 1, 3],
           [1, 1, 1])

b = Vector(3, 2, 2, 7, 4)



C = A.partition(Matrix(b).transpose()).gauss_jordan()
x = Vector(*[C[i][C.columns - 1] for i in range(C.rows)])

nullspace = A.homogeneous()
print(nullspace)
print(x)

