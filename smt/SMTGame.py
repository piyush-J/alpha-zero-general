from __future__ import print_function
import sys
sys.path.append('..')
from Game import Game
from SMTLogic import Board
import numpy as np
import glob, os
from z3 import * # may not need

"""
Game class implementation for SMT solving.
"""

class SMTGame(Game):
    def __init__(self, benchmarkPath = "example/", ext = "smt2"):
        self.bPath = benchmarkPath # now only for one formula; later change to benchmark
        os.chdir(self.bPath)
        self.formulaLst = []
        for f in glob.glob("*."+ ext):
            self.formulaLst.append(f)
        self.fSize = len(self.formulaLst)
        if self.fSize < 1: raise Exception("No smt file in the folder")
        self.nextFmID = 0

    def getInitBoard(self):
        bd = Board(self.formulaLst[self.nextFmID])
        if self.nextFmID == self.fSize - 1: self.nextFmID = 0
        else: self.nextFmID = self.nextFmID + 1
        return bd # return the board for now

    def getEmbedding(self, board):
        return board.get_state()

    def getActionSize(self, board):
        # return number of actions
        return len(board.get_legal_moves())

    def getNextState(self, board, action):
        # if takes action on board, return next board
        # action must be a valid move
        board.execute_move(action)
        return board
    #
    def getValidMoves(self, board):
        return board.get_legal_moves()

    def getGameEnded(self, board):
        # return 0 if not ended, 1 if solved, -1 if give up after certain number of attempts
        if board.is_win():
            return 1 # this can be related to time
        if board.is_giveup():
            return -1
        return 0 # relate to resources later
    #
    # def getCanonicalForm(self, board, player):
    #     # return state if player==1, else return -state if player==-1
    #     return player*board
    #
    # def getSymmetries(self, board, pi):
    #     # mirror, rotational
    #     assert(len(pi) == self.n**2+1)  # 1 for pass
    #     pi_board = np.reshape(pi[:-1], (self.n, self.n))
    #     l = []
    #
    #     for i in range(1, 5):
    #         for j in [True, False]:
    #             newB = np.rot90(board, i)
    #             newPi = np.rot90(pi_board, i)
    #             if j:
    #                 newB = np.fliplr(newB)
    #                 newPi = np.fliplr(newPi)
    #             l += [(newB, list(newPi.ravel()) + [pi[-1]])]
    #     return l
    #
    # def stringRepresentation(self, board):
    #     # 8x8 numpy array (canonical board)
    #     return board.tostring()
    #
    # @staticmethod
    # def display(board):
    #     n = board.shape[0]
    #
    #     print("   ", end="")
    #     for y in range(n):
    #         print (y,"", end="")
    #     print("")
    #     print("  ", end="")
    #     for _ in range(n):
    #         print ("-", end="-")
    #     print("--")
    #     for y in range(n):
    #         print(y, "|",end="")    # print the row #
    #         for x in range(n):
    #             piece = board[y][x]    # get the piece to print
    #             if piece == -1: print("X ",end="")
    #             elif piece == 1: print("O ",end="")
    #             else:
    #                 if x==n:
    #                     print("-",end="")
    #                 else:
    #                     print("- ",end="")
    #         print("|")
    #
    #     print("  ", end="")
    #     for _ in range(n):
    #         print ("-", end="-")
    #     print("--")
