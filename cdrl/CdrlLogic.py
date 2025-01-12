import numpy as np

class Board():

    def __init__(self, n=4, m=10, solver=None):
        "Set up initial board configuration."

        self.n = n # columns - actions (smaller)
        self.m = m # rows - episode length (bigger)
        # Create the empty board array with -1 filled
        self.pieces = [[-1]*self.n]*self.m # np.array([[-1]*4]*10) creates a 10x4 (mxn) array with -1 filled
        self.solver = solver
        assert self.solver is not None

    # add [][] indexer syntax to the Board
    def __getitem__(self, index): 
        return self.pieces[index]

    def get_legal_moves(self):
        
        current_state_literals = []
        
        for i in range(self.m):
            for j in range(self.n):
                if self.pieces[i][j] == 1:
                    # print(f"Value: {1}, i: {i}, j: {j}, pieces[i][j]: {self.pieces[i][j]}, i * n + j + 1: {i * self.n + j + 1}")
                    current_state_literals.append(i * self.n + j + 1)
                elif self.pieces[i][j] == 0:
                    # print(f"Value: {0}, i: {i}, j: {j}, pieces[i][j]: {self.pieces[i][j]}, -(i * n + j + 1): {-(i * self.n + j + 1)}")
                    current_state_literals.append(-(i * self.n + j + 1))
        
        # print("Current state literals:", current_state_literals)

        # Generate all possible next moves
        possible_moves = []
        for action in range(2**self.n):
            move = [int(x) for x in list(np.binary_repr(action, width=self.n))]
            possible_moves.append(move)
        
        # Check legality of each move
        legal_moves = []
        for move in possible_moves:
            new_pieces = [row[:] for row in self.pieces]
            self.execute_move(move, new_pieces)

            move_literals = []
            for i in range(self.m):
                for j in range(self.n):
                    if new_pieces[i][j] == 1:
                        move_literals.append(i * self.n + j + 1)
                    elif new_pieces[i][j] == 0:
                        move_literals.append(-(i * self.n + j + 1))

            assert self.solver is not None
            if self.solver.solve(assumptions=move_literals):
                legal_moves.append(move)

        return legal_moves # moves are in 2-D format (x,y)

    def has_legal_moves(self):
        # if self.pieces has -1 in last cell, then there is a legal move
        if -1 in self.pieces[-1]:
            return True
        else:
            return False
    
    def is_win(self):
        # TODO: arbitrary win condition for now, remove it later and call external tool within getGameEnded(), make sure it is transposed to 4x10
        # check whether there is a line of ones in the columns
        win = self.m
        for y in range(self.n):
            count = 0
            for x in range(self.m):
                if self[x][y]==1:
                    count += 1
            if count==win:
                return True
        
        return False
        
    def execute_move(self, move, pieces=None):
        # Use the provided pieces or default to self.pieces
        if pieces is None:
            pieces = self.pieces # in-place operation

        # Add a row to the board
        row_to_add = move
        # Replace the last non -1 filled row with the new row
        for i in range(len(pieces)):
            if -1 in pieces[i]:
                pieces[i] = row_to_add
                break
