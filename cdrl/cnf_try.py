from pysat.formula import CNF
from pysat.solvers import Solver
import numpy as np

m = 10
n = 4 

# read CNF file
cnf = CNF(from_file="../constraints10x4.cnf")

# Initialize solver with the CNF clauses
solver = Solver(bootstrap_with=cnf.clauses)

pieces = [[-1]*n]*m
pieces[0] = [1, 0, 0, 0]

current_state_literals = []
for i in range(m):
    for j in range(n):
        if pieces[i][j] == 1:
            print(f"Value: {1}, i: {i}, j: {j}, pieces[i][j]: {pieces[i][j]}, i * n + j + 1: {i * n + j + 1}")
            current_state_literals.append(i * n + j + 1)
        elif pieces[i][j] == 0:
            print(f"Value: {0}, i: {i}, j: {j}, pieces[i][j]: {pieces[i][j]}, -(i * n + j + 1): {-(i * n + j + 1)}")
            current_state_literals.append(-(i * n + j + 1))
        else: # unasigned
            print(f"Value: {-1}, i: {i}, j: {j}, pieces[i][j]: {pieces[i][j]}, i * n + j + 1: {i * n + j + 1}")

print("Current state literals:", current_state_literals)

# Function to execute a move
def execute_move(pieces, move):
    row_to_add = move
    for i in range(len(pieces)):
        if -1 in pieces[i]:
            pieces[i] = row_to_add
            break
    return pieces

# Generate all possible next moves
possible_moves = []
for action in range(2**n):
    move = [int(x) for x in list(np.binary_repr(action, width=n))]
    possible_moves.append(move)

# Check legality of each move
legal_moves = []
for move in possible_moves:
    new_pieces = [row[:] for row in pieces]
    new_pieces = execute_move(new_pieces, move)

    move_literals = []
    for i in range(m):
        for j in range(n):
            if new_pieces[i][j] == 1:
                move_literals.append(i * n + j + 1)
            elif new_pieces[i][j] == 0:
                move_literals.append(-(i * n + j + 1))

    if solver.solve(assumptions=move_literals):
        legal_moves.append(move)
        print("Legal Move:", move)
    else:
        print("Illegal Move:", move)
    
    # # Testing to add clause to negate the move_literals and check if the move is still legal (it should not be)
    # print("Move:", move)
    # print("before: ", solver.solve(assumptions=move_literals))
    # # add clause to negate the move_literals
    # solver.add_clause([-lit for lit in move_literals])
    # cnf.append([-lit for lit in move_literals])
    # print("after: ", solver.solve(assumptions=move_literals))
    # print("Clauses:", cnf.clauses)
    # # This part of the code can be used to add the clauses on the fly

# Print legal moves
print("Legal Moves:")
for move in legal_moves:
    print(move)

# # Convert move_literals to matrix representation
# move_matrix = [[-1 for _ in range(n)] for _ in range(m)]
# for lit in move_literals:
#     if lit > 0:
#         row = (lit - 1) // n
#         col = (lit - 1) % n
#         move_matrix[row][col] = 1
#     else:
#         row = (-lit - 1) // n
#         col = (-lit - 1) % n
#         move_matrix[row][col] = 0

# print("Move Matrix:")
# for row in move_matrix:
#     print(row)