
from calculator import *



A = Matrix([0, 2], 
            [1, 3])
P, L, U = A.plu()

print(P @ L @ U)