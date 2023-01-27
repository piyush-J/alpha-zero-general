from __future__ import print_function
import sys
sys.path.append('..')
from Game import Game
from .SMTLogic import CacheTreeNode
from .SMTLogic import Board
import numpy as np
import glob, os
import copy
from z3 import * # may not need

import functools
print = functools.partial(print, flush=True)

"""
Game class implementation for SMT solving.
"""

MODEL_OUT_FEATURES = 768
MANUAL_FEATURES = 5

WIN_REWARD = 1
NOCHANGE_REWARD = -1
FAIL_REWARD = -3
GIVEUP_REWARD = -1

STEP_WT = 0
TIME_WT = 0.00000002

class SMTGame(Game):
    def __init__(self, benchmarkPath, ext, moves_str, stats):
        self.bPath = benchmarkPath
        self.ext = ext
        self.formulaLst = []
        self.forest = [] # caching forest
        self.stats = stats
        self.moves_str = moves_str
        self.action_size = len(moves_str) # TODO: change later John: how
        for f in glob.glob(f"{self.bPath}/*.{self.ext}"):
            self.formulaLst.append(f)
            self.forest.append(CacheTreeNode(num_moves = self.action_size))
        self.fSize = len(self.formulaLst)
        if self.fSize < 1: raise Exception("No smt file in the folder")
        # self.curFmID = -1 # may not need
        self.nextFmID = 0
        self.accRlimit_all = [] # John: what's this?


    # def _make_representation(self): # TODO: smt
    #     return Board(self.formulaLst[self.curFmID], self.moves_str)

    def get_copy(self): # verified that a deep copy is not required for smt game
        return self
        # copy.deepcopy(self)

    def setNextFmID(self, id = 0): #set self.nextFmID value, and correspondingly, self.curFmID
        assert(id < self.fSize)
        self.nextFmID = id

    def getBenchmarkSize(self):
        return self.fSize

    # TODO: check whether need both currentID and nextID
    def getInitBoard(self):
        tnode = self.forest[self.nextFmID]
        bd = Board(self.nextFmID, self.formulaLst[self.nextFmID], self.moves_str, tnode, self.stats)
        # self.curFmID = self.nextFmID
        if self.nextFmID == self.fSize - 1: self.nextFmID = 0
        else: self.nextFmID = self.nextFmID + 1
        return bd # return the board

    def getManualEmbedding(self, board):
        return board.get_manual_state()

    def getEmbedding(self, board):
        return board.get_state()

    def getBoardSize(self):
        return MANUAL_FEATURES

    def getActionSize(self):
        # return number of actions
        # return len(board.get_legal_moves())
        return self.action_size # TODO: check if +1 is necessary

    # make sure with Piyush this won't be called on already ended board
    def getNextState(self, board, action):
        # if takes action on board, return next board
        # action must be a valid move
        new_board = board.execute_move(action)
        self.accRlimit_all.append(new_board.accRLimit)
        return new_board

    def getValidMoves(self, board):
        valids = [1]*self.getActionSize()
        _ = board.get_legal_moves() # just to check if the board is ended
        # print("legal moves: ", legal_moves)
        # if len(legal_moves)==0:
        #     valids[-1]=1
        #     return np.array(valids)
        # for x in legal_moves:
        # valids[x]=1
        return np.array(valids)

    def getGameEnded(self, board, level=0):
        # return 0 if not ended, 1 if solved, -1 if give up after certain number of attempts
        if board.is_fail():
            # print("fail")
            return FAIL_REWARD # - STEP_WT*level - TIME_WT*board.get_time()
        if board.is_nochange():
            # print("no_change; level: " + str(level) + " time: " + str(board.get_time()))
            return NOCHANGE_REWARD # - STEP_WT*level - TIME_WT*board.get_time()
        if board.is_win():
            # print("win")
            reward = WIN_REWARD - STEP_WT*level - TIME_WT*board.get_time()
            assert(reward > 0)
            return reward
        if board.is_giveup():
            # print("give up")
            return GIVEUP_REWARD # - STEP_WT*level - TIME_WT*board.get_time()
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
        return np.array2string(board.get_manual_state(), precision=2, separator=',',
                      suppress_small=True) + " " + str(board.is_done())
