'''
Board class for the game of TicTacToe.
Default board size is 3x3.
Board data:
  1=white(O), -1=black(X), 0=empty
  first dim is column , 2nd is row:
     pieces[0][0] is the top left square,
     pieces[2][0] is the bottom left square,
Squares are stored and manipulated as (x,y) tuples.

Author: Evgeny Tyurin, github.com/evg-tyurin
Date: Jan 5, 2018.

Based on the board for the game of Othello by Eric P. Nichols.

'''
from z3 import *
import numpy as np
from copy import copy

STEP_UPPER_BOUND = 2

# from bkcharts.attributes import color
class Board(): # Keep its name as Board for now; may call it goal later

    def __init__(self, formulaPath, moves_str):
        "Set up initial board configuration."

        self.fPath = formulaPath
        # print(formulaPath)
        self.moves_str = moves_str
        # Create the empty board array.
        self.formula = z3.parse_smt2_file(formulaPath) # maynot need to store this
        self.curGoal = z3.Goal()
        self.curGoal.add(self.formula)
        self.step = 0 # number of times a tactic as already been applied
        self.priorAction = []
        self.failed = False
    # Seems not relevant: let's see
    # add [][] indexer syntax to the Board
    # def __getitem__(self, index):
    #     return self.pieces[index]

    def __str__(self): # when you print board object
        return f"Embedding: {self.get_state()}; Current goal: {self.curGoal}; step: {self.step}; is_win: {self.is_win()}; is_giveup: {self.is_giveup()}"

    def is_done(self):
        return self.is_win() or self.is_giveup()

    def get_legal_moves(self):
        if not self.is_done():
            return set([i for i in range(len(self.moves_str))])
        else:
            return set()

    # Seems irrelevant
    # def has_legal_moves(self):
    #     for y in range(self.n):
    #         for x in range(self.n):
    #             if self[x][y]==0:
    #                 return True
    #     return False

    def get_state(self):
        p1 = Probe('size')
        numAssert = p1(self.curGoal)
        p2 = Probe('num-consts')
        numConst = p2(self.curGoal)
        return np.array([numAssert, numConst])

    def is_win(self):
        return (str(self.curGoal) == "[]") or (str(self.curGoal) == "[False]")

    def is_giveup(self):
        return self.step > STEP_UPPER_BOUND

    # current
    def execute_move(self, move):
        """Perform the given move on the board
        """
        # print(type(self.curGoal))
        result = copy(self)
        t = Tactic(self.moves_str[move])
        # print("Initial goal")
        # print(self.curGoal)
        output = t(self.curGoal)
        result.curGoal = z3.Goal()
        result.curGoal.add(output.as_expr()) # try outGoal[0]
        # print(type(self.curGoal))
        # print("after the move: " + move)
        # print(self.curGoal)
        result.step = result.step + 1

        result.priorAction.append(move)
        return result
