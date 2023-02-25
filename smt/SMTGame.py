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

# consider move these to json
# TOTAL_TIMEOUT = 180 # in seconds

MODEL_OUT_FEATURES = 768
MANUAL_FEATURES = 25 # change this later

MIN_SOLVED_REWARD = 0.1
WIN_REWARD = 1
STEP_EXCEED_REWARD = -0.5
LOSE_REWARD = -1
MAX_STEP = 9
# NOCHANGE_REWARD = -1
# FAIL_REWARD = -1
# GIVEUP_REWARD = -1

STEP_WT = 0
TIME_WT = 0.000000001

class SMTGame(Game):
    def __init__(self, benchmarkPath, ext, moves_str, stats, total_timeout, tactic_timeout, train = True):
        self.train = train
        self.bPath = benchmarkPath
        self.ext = ext
        self.formulaLst = []
        self.forest = [] # caching forest
        self.stats = stats
        self.total_timeout = total_timeout
        self.tactic_timeout = tactic_timeout
        self.moves_str = moves_str
        self.action_size = len(moves_str) # TODO: change later John: how
        assert(self.action_size > 1)
        for f in sorted(glob.glob(f"{self.bPath}/*.{self.ext}")):
            self.formulaLst.append(f)
            self.forest.append(CacheTreeNode(num_moves = self.action_size))
        self.fSize = len(self.formulaLst)
        self.solveLst = [False] *  self.fSize # recording whether a formula in the list has ever been solved
        if self.fSize < 1: raise Exception("No smt file in the folder")
        self.accRlimit_all = [] # John: what's this?

    # def _make_representation(self): # TODO: smt
    #     return Board(self.formulaLst[self.curFmID], self.moves_str)

    def get_copy(self): # verified that a deep copy is not required for smt game
        return self
        # copy.deepcopy(self)

    def getBenchmarkSize(self):
        return self.fSize

    def is_solvable(self, id):
        return self.solveLst[id]

    # need to change the overridden function signature in Game.py?
    def getInitBoard(self, id):
        assert(id < self.fSize) # may consider reiterate from the beginning when id >= size
        tnode = None
        if self.train: tnode = self.forest[id]
        bd = Board(id, self.formulaLst[id], self.moves_str, tnode, self.stats, self.total_timeout, self.train)
        return bd

    def getManualEmbedding(self, board):
        return board.get_manual_state()

        # this is currently used nowhere; check later
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
        new_board = board.execute_move(action, self.tactic_timeout)
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

    def getGameEnded(self, board):
        if board.is_win():
            self.solveLst[board.id] = True
            if board.get_time() > self.total_timeout: return MIN_SOLVED_REWARD
            return WIN_REWARD - (board.get_time()/self.total_timeout)*(WIN_REWARD - MIN_SOLVED_REWARD)
        if self.train:
            if board.step > MAX_STEP: return STEP_EXCEED_REWARD
        if board.is_timeout():
            return LOSE_REWARD
        return 0 # game not over yet

    def getCanonicalForm(self, board): # TODO
        """
        Input:
            board: current board

        Returns:
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
        return str(board.priorActions)
