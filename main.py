
from first import *

A = Matrix([1, 2, 3],
           [0, 4, 5],
           [1, 0, 6])

P, D, P_1 = A.diagonalize()
print(P @ D @ P_1)


