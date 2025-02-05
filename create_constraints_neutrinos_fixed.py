from pysat.formula import CNF
from pysat.solvers import Solver
import numpy as np

# ---------------------------
# Matrix dimensions
# ---------------------------
# TODO: some of the values are hardcoded, but they can be changed to be passed as arguments
m, n = 16, 8  # 16 rows, 8 columns
ROWS, COLS = m, n

# ---------------------------
# Helper function to convert variable number to string
# ---------------------------
def variable_to_string(var): # what does this variable represent?
    if var < 129:
        row = (var - 1) // 8 + 1
        col = (var - 1) % 8 + 1
        return f"Cell ({row}, {col})"
    elif var < 134:
        return f"act({var - 129 + 12})" # 129 -> 12, 133 -> 16
    elif var < 149:
        row = (var - 134) // 3 + 12
        k = (var - 134) % 3 + 1
        return f"vev({row}, {k})"
    elif var == 149:
        return "z2"
    elif var == 150:
        return "z3"
    else:
        return f"Unknown variable {var}"

# ---------------------------
# Helper functions for variable numbering
# ---------------------------
# Matrix variables: rows and columns are 1-indexed.
def var(i, j):
    assert 1 <= i <= m, "Row number must be in [1, m]."
    out = (i - 1) * n + j  # Rows 1..16, columns 1..8 => variables 1..128
    assert 1 <= out <= m * n, "Invalid variable number."
    return out

# ---------------------------
# Auxiliary variables for rows 12-16 (optional activity)
# ---------------------------
# We'll assign an auxiliary variable "act(r)" for each row r in 12..16.
# Since matrix variables are 1..128, let act(row) for row r be:
base_act = m * n  # 16*8 = 128
def act(r):
    assert 12 <= r <= 16, "Row number must be in [12, 16]."
    # r in [12, 16] mapped to numbers 129 to 133.
    out = base_act + (r - 11)  # row 12 -> 129, row 16 -> 133
    assert 129 <= out <= 133, "Invalid act variable number."
    return out

# ---------------------------
# Auxiliary variables for "vev" encoding in rows 12-16.
# For each row r (12..16), we use three variables: vev(r,1), vev(r,2), vev(r,3)
# Let these variables start after the act variables.
base_vev = base_act + (16 - 11)  # act variables occupy 129..133, so base_vev = 133
def vev_var(r, k):
    assert 12 <= r <= 16, "Row number must be in [12, 16]."
    assert 1 <= k <= 3, "k must be in [1, 3]."
    # r in [12,16], k in [1,3]. Returns a unique variable number.
    # For row 12 and k=1, variable is 134; row 12,k=2 is 135; …; row 16,k=3 is 134 + (5*3 - 1) = 148.
    out = base_vev + (r - 12) * 3 + k
    assert 134 <= out <= 148, "Invalid vev variable number."
    return out

# ---------------------------
# Additional auxiliary variables: z2 and z3
# They will be assigned numbers after the vev variables.
z2 = base_vev + 5 * 3 + 1     # 133 + 15 + 1 = 149
z3 = base_vev + 5 * 3 + 2  # 150

# ---------------------------
# Initialize CNF formula
# ---------------------------
cnf = CNF()

# ---------------------------
# Rows 1-3: Fixed rows
# ---------------------------
# Row 1:
# (a) Force columns 1-3 to 0.
for j in range(1, 4):
    cnf.append([-var(1, j)])
# (b) Force column 4 to 1.
cnf.append([var(1, 4)])
# (c) Exactly one among columns 5-8 is 1.
#     At least one:
cnf.append([var(1, 5), var(1, 6), var(1, 7), var(1, 8)])
#     At most one (pairwise):
for i1 in range(5, 9):
    for i2 in range(i1 + 1, 9):
        cnf.append([-var(1, i1), -var(1, i2)])

# Rows 2 and 3: Must be an exact copy of row 1.
for r in [2, 3]:
    for j in range(1, n + 1):
        x1 = var(1, j)
        xr = var(r, j)
        # x1 -> xr and xr -> x1:
        cnf.append([-x1, xr])
        cnf.append([x1, -xr])

# ---------------------------
# Rows 4-6: Choice rows with inter-row condition (using row 4 as base)
# ---------------------------
for r in [4, 5, 6]:
    # (a) Exactly one among the first 4 columns:
    clause = [var(r, j) for j in range(1, 5)]
    cnf.append(clause)
    for i1 in range(1, 5):
        for i2 in range(i1 + 1, 5):
            cnf.append([-var(r, i1), -var(r, i2)])
    # (b) Exactly one among the last 4 columns:
    clause = [var(r, j) for j in range(5, 9)]
    cnf.append(clause)
    for i1 in range(5, 9):
        for i2 in range(i1 + 1, 9):
            cnf.append([-var(r, i1), -var(r, i2)])

# Inter-row condition for rows 4-6:
# Using row 4 as the base row:
for r in [5, 6]:
    for j in range(1, n + 1):
        # If row 4's col4 is 1 then enforce row4[j] <-> row r[j]:
        cnf.append([-var(4, 4), -var(4, j), var(r, j)])
        cnf.append([-var(4, 4), -var(r, j), var(4, j)])
for r in [5, 6]:
    # If row 4's col4 is 0 then rows 5 and 6 must have col4 = 0.
    cnf.append([var(4, 4), -var(r, 4)])

# ---------------------------
# Rows 7-9: Same rules as rows 4-6 (using row 7 as base)
# ---------------------------
for r in [7, 8, 9]:
    # (a) Exactly one among the first 4 columns:
    clause = [var(r, j) for j in range(1, 5)]
    cnf.append(clause)
    for i1 in range(1, 5):
        for i2 in range(i1 + 1, 5):
            cnf.append([-var(r, i1), -var(r, i2)])
    # (b) Exactly one among the last 4 columns:
    clause = [var(r, j) for j in range(5, 9)]
    cnf.append(clause)
    for i1 in range(5, 9):
        for i2 in range(i1 + 1, 9):
            cnf.append([-var(r, i1), -var(r, i2)])

# Inter-row condition for rows 7-9 (using row 7 as base):
for r in [8, 9]:
    for j in range(1, n + 1):
        cnf.append([-var(7, 4), -var(7, j), var(r, j)])
        cnf.append([-var(7, 4), -var(r, j), var(7, j)])
for r in [8, 9]:
    cnf.append([var(7, 4), -var(r, 4)])

# ---------------------------
# Rows 10-11: Fixed pattern rows
# ---------------------------
for r in [10, 11]:
    # (a) First element must be 1.
    cnf.append([var(r, 1)])
    # (b) Columns 2-4 must be 0.
    for j in [2, 3, 4]:
        cnf.append([-var(r, j)])
    # (c) Exactly one among columns 5-8 is 1.
    cnf.append([var(r, 5), var(r, 6), var(r, 7), var(r, 8)])
    for i1 in range(5, 9):
        for i2 in range(i1 + 1, 9):
            cnf.append([-var(r, i1), -var(r, i2)])

# ---------------------------
# Rows 12-16: Optional activity with cascading zero rows + vev encoding
# ---------------------------
for r in range(12, 17):
    # --- Optional activity for row r ---
    # (i) If row r is NOT active then every cell must be 0.
    for j in range(1, n + 1):
        # (act(r) OR not var(r,j))
        cnf.append([act(r), -var(r, j)])
    # (ii) If row r IS active then:
    #      - Exactly one among the first 4 columns:
    clause = [-act(r)] + [var(r, j) for j in range(1, 5)]
    cnf.append(clause)
    for i1 in range(1, 5):
        for i2 in range(i1 + 1, 5):
            cnf.append([-act(r), -var(r, i1), -var(r, i2)])
    #      - Exactly one among the last 4 columns:
    clause = [-act(r)] + [var(r, j) for j in range(5, 9)]
    cnf.append(clause)
    for i1 in range(5, 9):
        for i2 in range(i1 + 1, 9):
            cnf.append([-act(r), -var(r, i1), -var(r, i2)])
    
    # --- vev encoding for row r ---
    # Let x be the 4th element in row r.
    # For each k in {1,2,3}, if x is 0 then vev(r,k) must be 0.
    for k in range(1, 4):
        cnf.append([var(r, 4), -vev_var(r, k)])  # x OR not vev
    # If x is 1 then at least one of the vev variables must be 1.
    cnf.append([-var(r, 4), vev_var(r, 1), vev_var(r, 2), vev_var(r, 3)])
    # And if x is 1 then at most one of them can be 1.
    for k1 in range(1, 4):
        for k2 in range(k1 + 1, 4):
            cnf.append([-var(r, 4), -vev_var(r, k1), -vev_var(r, k2)])

# Cascading condition: if any row r (for r=12..15) is zero then every subsequent row is zero.
for r in range(12, 16):
    # If not active in row r then row r+1 must also not be active:
    cnf.append([act(r), -act(r + 1)])

# ---------------------------
# (z2 and z3 are free auxiliary variables; no extra clauses are needed.)
# ---------------------------
# Their variable numbers are z2 (149) and z3 (150).

# ---------------------------
# save CNF formula to a CNF file
# ---------------------------
cnf.to_file(f"constraints{m}x{n}_neutrinos_fixed.cnf")

# ---------------------------
# Solving the CNF and outputting the result
# ---------------------------
# with Solver(bootstrap_with=cnf.clauses) as solver:
#     if solver.solve():
#         model = solver.get_model()  # Model is a list of assigned integers.
        
#         # Build the 16x8 solution matrix for the original variables.
#         matrix = np.zeros((m, n), dtype=int)
#         for i in range(1, m + 1):
#             for j in range(1, n + 1):
#                 matrix[i - 1, j - 1] = 1 if var(i, j) in model else 0

#         print("Solution matrix (rows 1-16, columns 1-8):")
#         print(matrix)
        
#         # Optionally, print the assignments for the auxiliary variables.
#         # Print act variables for rows 12-16:
#         print("\nRow activity (act) for rows 12-16:")
#         for r in range(12, 17):
#             print(f"Row {r} active: {1 if act(r) in model else 0}")
        
#         # Print vev assignments for rows 12-16:
#         print("\nvev assignments for rows 12-16 (one-hot for value in {1,2,3}, 0 means all off):")
#         for r in range(12, 17):
#             vev_assignment = [1 if vev_var(r, k) in model else 0 for k in range(1, 4)]
#             print(f"Row {r} vev: {vev_assignment}")
        
#         # Print the values of z2 and z3:
#         print("\nz2 =", 1 if z2 in model else 0, ", z3 =", 1 if z3 in model else 0)
        
#     else:
#         print("No solution found.")
