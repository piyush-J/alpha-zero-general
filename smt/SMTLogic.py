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

# from bkcharts.attributes import color
class Board(): # Keep its name as Board for now; may call it goal later

    def __init__(self, formulaPath):
        "Set up initial board configuration."

        self.fPath = formulaPath
        # Create the empty board array.
        self.formula = z3.parse_smt2_file(formulaPath)
        self.curGoal = z3.Goal()
        self.curGoal.add(self.formula)
        self.step = 0 # number of times a tactic as already been applied
    # Seems not relevant: let's see
    # add [][] indexer syntax to the Board
    # def __getitem__(self, index):
    #     return self.pieces[index]

    def get_legal_moves(self):
        return ["simplify", "smt"]

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
        return [numAssert, numConst]

    def is_win(self):
        # print(str(self.curGoal))
        result = str(self.curGoal) == "[]"
        # print(result)
        return result

    def is_giveup(self):
        return self.step > 2 # change this number to a constant variable later

    # current 
    def execute_move(self, move):
        """Perform the given move on the board;
        """
        # print(type(self.curGoal))
        t = Tactic(move)
        # print("Initial goal")
        # print(self.curGoal)
        outGoal = t(self.curGoal)
        self.curGoal = z3.Goal()
        self.curGoal.add(outGoal.as_expr())
        # print(type(self.curGoal))
        # print("after the move: " + move)
        # print(self.curGoal)
        self.step = self.step + 1
