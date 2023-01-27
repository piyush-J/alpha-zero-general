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
import copy

STEP_UPPER_BOUND = 8
TACTIC_TIMEOUT = 10000 # in milliseconds

def get_rlimit(tmpSolver):
    stats = tmpSolver.statistics()
    for i in range(len(stats)):
        if stats[i][0] == 'rlimit count':
            return stats[i][1]
    return 0

#TO_DO: think more about whether can store/return reference/copy; currently store as reference, return a copy
class CacheTreeNode():
    def __init__(self, num_moves, bd = None):
        self.board = bd # for original formula no need to cache the board; for root of every tree in the forest, this field is None
        self.numMoves = num_moves
        self.childLst = [None] * self.numMoves

    # def updateChild(self, resBoard, move):
    #     assert(childLst[move] == None)
    #     treeNode = CacheTreeNode(resBoard, self.numMoves)
    #     self.childLst[move] = treeNode

class Board(): # Keep its name as Board for now; may call it goal later
    def __init__(self, ID, formulaPath, moves_str, cTreeNode, stats, train):
        "Set up initial board configuration."
        self.train = train
        self.id = ID
        self.fPath = formulaPath
        self.cacheTN = cTreeNode
        # print(formulaPath)
        self.moves_str = moves_str
        self.stats = stats
        # Create the empty board array.
        self.formula = z3.parse_smt2_file(formulaPath) # maynot need to store this
        self.initGoal = z3.Goal()
        self.initGoal.add(self.formula)
        self.curGoal = self.initGoal
        self.step = 0 # number of times tactics have already been applied
        self.priorActions = [] # store list of tactic strings
        self.failed = False
        self.nochange  = False
        self.accRLimit = 0 # machine-independent timing
        self.rlimit = None # rlimit for executing the last tactic

    def __str__(self): # when you print board object
        return f"fID: {self.id}; fPath: {self.fPath}; Embedding: {self.get_manual_state()}; step: {self.step}; is_win: {self.is_win()}; is_nochange: {self.is_nochange()}; is_fail: {self.is_fail()}; accRLimit: {self.accRLimit}"

    def get_legal_moves(self): # Not required for this game, but may be useful for other games
        if self.is_done():
            raise Exception("Game is already over")

    def get_state(self):
        return str(self.curGoal)

    def get_manual_state(self):
        p1 = Probe('num-consts')
        p2 = Probe('num-exprs')
        p3 = Probe('size')
        # p4 = Probe('is-qfbv-eq')
        p5 = Probe('is-unbounded')
        p6 = Probe('is-pb')
        numConst = p1(self.curGoal)
        numConst = (numConst-self.stats["num_consts"][0])/(self.stats["num_consts"][1]-self.stats["num_consts"][0])
        numExpr = p2(self.curGoal)
        numExpr = (numExpr-self.stats["num_exprs"][0])/(self.stats["num_exprs"][1]-self.stats["num_exprs"][0])
        numAssert = p3(self.curGoal)
        numAssert = (numAssert-self.stats["size"][0])/(self.stats["size"][1]-self.stats["size"][0])
        isUnbound = p5(self.curGoal)
        isPB = p6(self.curGoal)

        priorActionsInt = [self.moves_str.index(act)+1 for act in self.priorActions] # +1 to avoid 0 (0 is reserved for padding)
        prior_actions_padded = priorActionsInt + [0] * (STEP_UPPER_BOUND - len(priorActionsInt) + 1)

        return np.array([numConst, numExpr, numAssert, isUnbound, isPB] + prior_actions_padded)

    def get_time(self):
        return self.accRLimit

    def is_win(self):
        return (str(self.curGoal) == "[]") or (str(self.curGoal) == "[False]")

    def is_fail(self):
        return self.failed

    def is_nochange(self):
        return self.nochange

    def is_giveup(self):
        return self.step > STEP_UPPER_BOUND

    def is_done(self):
        if self.train:
            return self.is_win() or self.is_giveup() or self.is_fail() or self.is_nochange()
        return self.is_win()

    # with the current caching design, timeout cannot be changed for a formula
    def transformNextState(self, move, timeout):
        if self.train:
            self.cacheTN = CacheTreeNode(len(self.moves_str),self) # may relate to action size?
        # print(self.moves_str[move])
        t = Tactic(self.moves_str[move])
        self.priorActions.append(self.moves_str[move])
        # self.priorMoves.append(move)
        tTimed = TryFor(t, TACTIC_TIMEOUT)
        prevGoalStr = str(self.curGoal)
        tmp = z3.Solver()
        rlimit_before = get_rlimit(tmp)
        try:
            tResult = tTimed(self.curGoal)
            assert(len(tResult) == 1)
            self.curGoal = tResult[0]
            if prevGoalStr == str(self.curGoal):
                self.nochange = True
        except Z3Exception:
            self.failed = True
        rlimit_after = get_rlimit(tmp)
        self.rlimit = rlimit_after - rlimit_before
        self.accRLimit += self.rlimit
        self.step = self.step + 1


    def execute_move(self, move, timeout):
        """Perform the given move on the board and return the result board
        """
        assert(not self.is_done())
        if (self.train) and (self.cacheTN.childLst[move] is not None):
            return self.cacheTN.childLst[move].board
        # print("no cache")
        result = copy.deepcopy(self)
        result.transformNextState(move, timeout)
        # remember to update the cahce tree
        if self.train:
            self.cacheTN.childLst[move] = result.cacheTN
        return result
