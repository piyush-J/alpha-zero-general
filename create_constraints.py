from pysat.formula import CNF
from pysat.solvers import Solver
import numpy as np

# Fill in row-wise for the m x n matrix, e.g., for m = 4, n = 2: [[var1, var2], [var3, var4], [var5, var6], [var7, var8]]

# Initialize CNF formula
cnf = CNF()

m = 10
n = 4

# Add clauses to ensure each row has exactly one cell assigned as 1
for i in range(m):
    # At least one variable in the row must be true
    cnf.append([i * n + j + 1 for j in range(n)])
    
    # No two variables in the same row can be true simultaneously
    for j in range(n):
        for k in range(j + 1, n):
            cnf.append([-(i * n + j + 1), -(i * n + k + 1)])

# Print the CNF clauses
# print("CNF Clauses:")
# for clause in cnf.clauses:
#     print(clause)

# save the CNF formula to a CNF file
cnf.to_file(f"constraints{m}x{n}.cnf")