from __future__ import print_function
import sys
sys.path.append('..')
from Game import Game
from .CdrlLogic import Board
import numpy as np

class CdrlGame(Game):
    def __init__(self, n=3):
        self.n = n 

    def getInitBoard(self):
        # return initial board (numpy board)
        b = Board(self.n)
        return np.array(b.pieces)

    def getBoardSize(self):
        # (a,b) tuple
        return (self.n, self.n)

    def getActionSize(self):
        # return number of actions
        # return self.n*self.n + 1 # 9x9 cells + 1 pass

        # 2^n possible moves
        return 2**self.n + 1 # [0,0,0], [0,0,1], .., [1,1,1] + 1 invalid move indicator

    def getNextState(self, board, action):
        # if player takes action on board, return next (board,player)
        # action must be a valid move
        b = Board(self.n)
        b.pieces = np.copy(board)
        move = [int(x) for x in list(np.binary_repr(action, width=self.n))] # convert 0 to [0,0,0], 1 to [0,0,1], .., 7 to [1,1,1]
        b.execute_move(move)
        return b.pieces

    def getValidMoves(self, board):
        # return a fixed size binary vector
        valids = [0]*self.getActionSize()
        b = Board(self.n)
        b.pieces = np.copy(board)
        legalMoves =  b.get_legal_moves() 
        if len(legalMoves)==0:
            valids[-1]=1 
            return np.array(valids)
        # convert move (2-D) to action (1-D) [0,0,0] to 0, [0,0,1] to 1, .., [1,1,1] to 7, etc.
        for x in legalMoves:
            action = int(''.join(map(str, x)), 2)
            valids[action]=1
        return np.array(valids)

    def getGameEnded(self, board):
        #TODO: check to-do in is_win() function -> change the return values to the ones you obtain from the external tool + ingest the conflict clauses from the external tool
        # return 0 if not ended, 1 if win, -1 if lose
        b = Board(self.n)
        b.pieces = np.copy(board)

        if b.is_win():
            return 1
        if b.has_legal_moves():
            return 0
        else: # lose
            return -1

    def getCanonicalForm(self, board):
        return board

    def getSymmetries(self, board, pi): 
        # TODO: implement symmetries?
        return [(board, pi)]

    def stringRepresentation(self, board):
        return '\n'.join([' '.join(map(str, board[:, col])) for col in range(board.shape[1])])

    @staticmethod
    def display(board):
        print(board)
