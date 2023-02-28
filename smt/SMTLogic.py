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
import time

# consider move these to json
# TOTAL_TIMEOUT = 180 # in seconds
PREV_ACTIONS_EMBED = 5 # number of previous actions you want to include in your embedding for prior actions
# STEP_UPPER_BOUND = 8 # upper bound on the number of steps to take in a game
# TACTIC_TIMEOUT = 10000 # in milliseconds

def get_rlimit(tmpSolver):
    stats = tmpSolver.statistics()
    for i in range(len(stats)):
        if stats[i][0] == 'rlimit count':
            return stats[i][1]
    return 0

def toSMT2Benchmark(f, status="unknown", name="benchmark", logic=""):
    v = (Ast * 0)()
    return Z3_benchmark_to_smtlib_string(f.ctx_ref(), name, logic, status, "", 0, v, f.as_ast())

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
    def __init__(self, ID, formulaPath, moves_str, cTreeNode, stats, total_timeout, train):
        "Set up initial board configuration."
        self.train = train
        self.id = ID
        self.fPath = formulaPath
        self.cacheTN = cTreeNode
        # print(formulaPath)
        self.moves_str = moves_str
        self.stats = stats
        self.total_timeout = total_timeout
        # Create the empty board array.
        # self.cntx = Context()
        # self.formula = z3.parse_smt2_file(formulaPath, ctx=self.cntx)
        # self.initGoal = z3.Goal(ctx=self.cntx)
        # self.initGoal.add(self.formula)
        with open(self.fPath, 'r') as f: #may write to a temp file later
            self.curGoal = f.read()
        self.step = 0 # number of times tactics have already been applied
        self.priorActions = [] # store list of tactic strings
        self.win = False
        self.failed = False
        self.nochange  = False
        self.accRLimit = 0 # machine-independent time
        self.accTime = 0 # time
        self.rlimit = None # rlimit for executing the last tactic

    def __str__(self): # when you print board object
        return f"fID: {self.id}; fPath: {self.fPath}; Embedding: {self.get_manual_state()}; step: {self.step}; is_win: {self.is_win()}; is_nochange: {self.is_nochange()}; is_fail: {self.is_fail()}; accTime: {self.accTime}, accRLimit: {self.accRLimit}"

    # mixed shallow&deep copy
    def get_copy(self):
        copied = copy.copy(self)
        # it is not important how to take care of self.cacheTN as it's taken care of in transformNextState()
        copied.priorActions = copy.copy(copied.priorActions)
        return copied

    def get_legal_moves(self): # Not required for this game, but may be useful for other games
        if self.is_done():
            raise Exception("Game is already over")

    # John: when does this used?
    def get_state(self):
        return self.curGoal

    # currently +1 so that 0 represent no prior action
    def get_most_recent_action(self):
        if len(self.priorActions) == 0: return 0
        return self.moves_str.index(self.priorActions[-1])+1

    def get_manual_state(self):
        probeStrLst = ['is-unbounded', 'is-pb', 'arith-max-deg', 'arith-avg-deg', 'arith-max-bw', 'arith-avg-bw', 'is-qflia',
            'is-qflra', 'is-qflira', 'is-qfnia', 'is-qfnra', 'memory', 'depth', 'size', 'num-exprs', 'num-consts', 'num-bool-consts', 'num-arith-consts', 'num-bv-consts', 'has-quantifiers',
            'has-patterns', 'is-propositional', 'is-qfbv', 'is-qfaufbv', 'is-quasi-pb']
        measureLst = []
        cntx = z3.Context()
        formula = z3.parse_smt2_string(self.curGoal, ctx=cntx)
        goal = z3.Goal(ctx=cntx)
        goal.add(formula)
        for pStr in probeStrLst:
            p = z3.Probe(pStr, ctx=cntx)
            measure = p(goal)
            if pStr in self.stats:
                measure = (measure-self.stats[pStr][0])/(self.stats[pStr][1]-self.stats[pStr][0])
            measureLst.append(measure)
        priorActionsInt = [self.moves_str.index(act)+1 for act in self.priorActions] # +1 to avoid 0 (0 is reserved for padding)
        priorActionsInt = priorActionsInt[-(PREV_ACTIONS_EMBED):] # only keep the last PREV_ACTIONS_EMBED actions
        prior_actions_padded = priorActionsInt + [0] * (PREV_ACTIONS_EMBED - len(priorActionsInt))
        return np.array(measureLst + prior_actions_padded)

    def get_time(self):
        return self.accTime

    def is_win(self):
        return self.win

    def is_fail(self):
        return self.failed

    def is_nochange(self):
        return self.nochange

    def is_giveup(self):
        return self.step > STEP_UPPER_BOUND

    # episode timeout
    def is_timeout(self):
        return self.accTime > self.total_timeout

    # merge with getGameEnded later
    def is_done(self):
        return self.is_win() or self.is_timeout()

    # TO_DO: currently no_change or fail just mean in the process it has happened sometime
    # with the current caching design, timeout cannot be changed for a formula
    def transformNextState(self, move, timeout):
        if self.train:
            self.cacheTN = CacheTreeNode(len(self.moves_str),self) # may relate to action size?
        cntx = z3.Context()
        tmp = z3.Solver(ctx = cntx)
        rlimit_before = get_rlimit(tmp)
        time_before = time.time()
        formula = z3.parse_smt2_string(self.curGoal, ctx=cntx)
        pre_goal = z3.Goal(ctx=cntx)
        pre_goal.add(formula)
        t = Tactic(self.moves_str[move], ctx = cntx)
        self.priorActions.append(self.moves_str[move])
        # self.priorMoves.append(move)
        tTimed = TryFor(t, timeout * 1000)
        try:
            new_goals = tTimed(pre_goal)
            assert(len(new_goals) == 1)
            new_goal = new_goals[0]
            if (str(new_goal) == "[]") or (str(new_goal) == "[False]"): self.win = True
            if str(pre_goal) == str(new_goal):
                self.nochange = True
            else:
                e = new_goal.as_expr()
                self.curGoal = toSMT2Benchmark(e, logic="QF_NIA")
        except Z3Exception:
            self.failed = True
        rlimit_after = get_rlimit(tmp)
        time_after = time.time()
        self.rlimit = rlimit_after - rlimit_before
        self.accTime += (time_after - time_before)
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
