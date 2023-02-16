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

PREV_ACTIONS_EMBED = 3 # number of previous actions you want to include in your embedding for prior actions
STEP_UPPER_BOUND = 8 # upper bound on the number of steps to take in a game
# TACTIC_TIMEOUT = 10000 # in milliseconds

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
        self.cntx = Context()
        self.formula = z3.parse_smt2_file(formulaPath, ctx=self.cntx) # maynot need to store this
        self.initGoal = z3.Goal(ctx=self.cntx)
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

    # mixed shallow&deep copy
    def get_copy(self):
        copied = copy.copy(self)
        # it is not important how to take care of self.cacheTN as it's taken care of in transformNextState()
        copied.curGoal = copy.deepcopy(copied.curGoal)
        copied.priorActions = copy.copy(copied.priorActions)
        return copied

    def get_legal_moves(self): # Not required for this game, but may be useful for other games
        if self.is_done():
            raise Exception("Game is already over")

    def get_state(self):
        return str(self.curGoal)

    # currently +1 so that 0 represent no prior action
    def get_most_recent_action(self):
        if len(self.priorActions) == 0: return 0
        return self.moves_str.index(self.priorActions[-1])+1

    def get_manual_state(self):
        probeStrLst = ['is-unbounded', 'is-pb', 'arith-max-deg', 'arith-avg-deg', 'arith-max-bw', 'arith-avg-bw', 'is-qflia',
            'is-qflra', 'is-qflira', 'is-qfnia', 'is-qfnra', 'memory', 'depth', 'size', 'num-exprs', 'num-consts', 'num-bool-consts', 'num-arith-consts', 'num-bv-consts', 'has-quantifiers',
            'has-patterns', 'is-propositional', 'is-qfbv', 'is-qfaufbv', 'is-quasi-pb']
        measureLst = []
        for pStr in probeStrLst:
            p = Probe(pStr)
            measure = p(self.curGoal)
            if pStr in self.stats:
                measure = (measure-self.stats[pStr][0])/(self.stats[pStr][1]-self.stats[pStr][0])
            measureLst.append(measure)
        priorActionsInt = [self.moves_str.index(act)+1 for act in self.priorActions] # +1 to avoid 0 (0 is reserved for padding)
        priorActionsInt = priorActionsInt[-(PREV_ACTIONS_EMBED):] # only keep the last PREV_ACTIONS_EMBED actions
        prior_actions_padded = priorActionsInt + [0] * (PREV_ACTIONS_EMBED - len(priorActionsInt))
        return np.array(measureLst + prior_actions_padded)

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
        return self.is_win() or self.is_giveup()

    # with the current caching design, timeout cannot be changed for a formula
    def transformNextState(self, move, timeout):
        if self.train:
            self.cacheTN = CacheTreeNode(len(self.moves_str),self) # may relate to action size?
        # print(self.moves_str[move])
        t = Tactic(self.moves_str[move], ctx = self.cntx)
        self.priorActions.append(self.moves_str[move])
        # self.priorMoves.append(move)
        tTimed = TryFor(t, timeout * 1000)
        prevGoalStr = str(self.curGoal)
        tmp = z3.Solver(ctx = self.cntx)
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
        result = self.get_copy()
        result.transformNextState(move, timeout)
        # remember to update the cahce tree
        if self.train:
            self.cacheTN.childLst[move] = result.cacheTN
        return result
