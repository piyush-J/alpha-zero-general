from __future__ import print_function
import sys
sys.path.append('..')
from Game import Game
from .SMTLogic import Board
import numpy as np
from z3 import * # may not need

"""
Game class implementation for SMT solving.
"""

class SMTGame(Game):
    def __init__(self, formulaPath = "smt/example/qfniaex.smt2", moves_str=("simplify", "smt")):
        self.fPath = formulaPath # now only for one formula; later change to benchmark
        self.moves_str = moves_str
        self.action_size = len(moves_str) # TODO: change later

    def _make_representation(self): # TODO: smt
        return Board(self.fPath, self.moves_str)

    def get_copy(self): # TODO: check if this is necessary later
        return SMTGame(self.fPath, self.moves_str)

    def getInitBoard(self):
        bd = Board(self.fPath, self.moves_str) # store as a member for now
        return bd # return the board for now - made all changes accordingly - donot change

    def getEmbedding(self, board):
        return board.get_state()

    def getBoardSize(self):#, board): 
        # return len(board.get_state())
        return 2

    def getActionSize(self):
        # return number of actions
        # return len(board.get_legal_moves()) 
        return self.action_size + 1 # TODO: check if +1 is necessary

    def getNextState(self, board, action):
        # if takes action on board, return next board
        # action must be a valid move
        b2 = self._make_representation()
        b2.curGoal = board.curGoal
        b2.step = board.step
        b2.execute_move(action) # we need to execute the move on a copy otherwise the original board will be changed in MCTS recursion
        return b2
    
    def getValidMoves(self, board):
        valids = [0]*self.getActionSize()
        legal_moves = board.get_legal_moves()
        # print("legal moves: ", legal_moves)
        if len(legal_moves)==0:
            valids[-1]=1
            return np.array(valids)
        for x in legal_moves:
            valids[x]=1
        return np.array(valids)

    def getGameEnded(self, board):
        # return 0 if not ended, 1 if solved, -1 if give up after certain number of attempts
        if board.is_win():
            return 1 # this can be related to time
        if board.is_giveup():
            return -5
        return 0 # relate to resources later # game not over yet
    
    def getCanonicalForm(self, board): # TODO
        """
        Input:
            board: current board

        Returns:
            canonicalBoard: returns canonical form of board. The canonical form
                            should be independent of player. For e.g. in chess,
                            the canonical form can be chosen to be from the pov
                            of white. When the player is white, we can return
                            board as is. When the player is black, we can invert
                            the colors and return the board.
        """
        return board

    def getSymmetries(self, board, pi): # TODO
        """
        Input:
            board: current board
            pi: policy vector of size self.getActionSize()

        Returns:
            symmForms: a list of [(board,pi)] where each tuple is a symmetrical
                       form of the board and the corresponding pi vector. This
                       is used when training the neural network from examples.
        """
        return [(board, pi)]

    def stringRepresentation(self, board): # TODO
        """
        Input:
            board: current board

        Returns:
            boardString: a quick conversion of board to a string format.
                         Required by MCTS for hashing.
        """
        # return board.tobytes()s
        return ",".join([str(i) for i in board.get_state()])
