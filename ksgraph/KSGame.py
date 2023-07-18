import copy
from Game import Game
from .KSLogic import Board
from .KSLogicMode0 import BoardMode0

import numpy as np
from pysat.formula import CNF

import networkx as nx
import wandb
import matplotlib.pyplot as plt

import hashlib

def calculate_hash(string):
    sha256_hash = hashlib.sha256()
    sha256_hash.update(string.encode('utf-8'))
    return sha256_hash.hexdigest()

class KSGame(Game):
    def __init__(self, args, filename): 
        super(KSGame, self).__init__()
        self.args = args
        self.cnf = CNF(from_file=filename)
        self.order = args.order
        self.MAX_LITERALS = args.MAX_LITERALS
        self.STATE_SIZE = args.STATE_SIZE
        print("Solving the KS system of order ", self.order, " with ", len(self.cnf.clauses), " constraints")

        self.log_sat_asgn = []
        self.log_giveup_rew, self.log_giveup_rewA = [], []
        self.log_eval_var, self.log_eval_varA = [], []

        self.edge_dict = {}
        # self.tri_dict = {}
        count = 0
        for j in range(1, self.order+1):             #generating the edge variables
            for i in range(1, self.order+1):
                if i < j:
                    count += 1
                    self.edge_dict[(i,j)] = count

        self.edge_count = count
        assert self.MAX_LITERALS >= count # sanity check so that we can encode the edge variables in the action space

        # for a in range(1, order-1):             #generating the triangle variables
        #     for b in range(a+1, order):
        #         for c in range(b+1, order+1):
        #             count += 1
        #             self.tri_dict[(a,b,c)] = count

    def _make_representation(self):
        if self.args.MCTSmode in [0, 2]:
            board = BoardMode0(args=self.args, cnf=self.cnf, edge_dict=self.edge_dict, max_metric_val=len(self.cnf.clauses))
            board.calculate_march_metrics() # initialize the valid literals, prob, and march_var_score_dict
            return board
        return Board(args=self.args, cnf=self.cnf, edge_dict=self.edge_dict)

    def get_copy(self):
        return self # copy.deepcopy(self) # TODO: check if deepcopy is required

    def getInitBoard(self):
        bd = self._make_representation()
        return bd

    def getEmbedding(self, board):
        return board.get_state()

    def getBoardSize(self): # used by NN
        return self.STATE_SIZE

    def getActionSize(self): 
        return int(self.MAX_LITERALS*2 + 1) # [e.g., dummy_0, 1 to 10, -1 to -10]
    
    def getNv(self): # no. of variables
        return self.cnf.nv

    def getNextState(self, board, action):
        assert action > 0, "Invalid action"
        new_board = board.execute_move(action)
        return new_board

    def getValidMoves(self, board):
        assert not board.is_done()
        valids = [0]*self.getActionSize()
        legalMoves =  [board.lits2var[l] for l in board.get_legal_literals()]
        for x in legalMoves:
            valids[x]=1
        return np.array(valids)

    def getGameEnded(self, board: Board, eval_cls=False):
        """
        Input:
            board: current board

        Returns:
            r: 0 if game has not ended, reward otherwise. 
               
        """
        if self.args.MCTSmode in [0, 2]:
            return self.getGameEndedMode0(board, eval_cls)

        if board.is_done(): #TODO: include is_unknown()
            rew, solver_model = board.compute_reward()
            assert rew is not None
            if board.is_win(): # sat
                assert solver_model is not None
                self.log_sat_asgn.append(solver_model)
            elif board.is_fail():
                assert solver_model is None
            elif board.is_giveup():
                if board.arena_mode: # separate logging for arena mode
                    self.log_giveup_rewA.append(rew)
                    self.log_eval_varA.append(board.total_rew)
                else:
                    self.log_giveup_rew.append(rew)
                    self.log_eval_var.append(board.total_rew)
                if solver_model is not None: # sat determined by sat solver
                    self.log_sat_asgn.append(solver_model) # already includes the prior actions (assumptions)
            return rew
        else:
            return None
        
    def getGameEndedMode0(self, board, eval_cls):
        if board.is_done():
            return board.compute_reward(eval_cls)
        else:
            return None

    def getCanonicalForm(self, board):
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

    # def _unique_permutations(x):
    #     # ys = list of (row_idx, boolean) indicating whether 
    #     # or not the row at index idx has non-zero elements
    #     ys = list(zip(range(len(x)), np.any(x != 0, axis=1))) 
    #     # sort the list so that the last elemant can be (row_idx, False)
    #     # IF there is a row with only zeros
    #     ys = sorted(ys, key=lambda x: x[1], reverse=True)
    #     # keep the idx if (idx, True) else keep the idx of the last element
    #     idxs = [x[0] if x[1] else ys[-1][0] for x in ys]
    #     # compute the permutations without duplicates and map back to the input
    #     return [x[p] for p in multiset_permutations(idxs)]

    def getSymmetries(self, board, pi):
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

    def stringRepresentation(self, board):
        """
        Input:
            board: current board

        Returns:
            boardString: a quick conversion of board to a string format.
                         Required by MCTS for hashing.
        """
        return str(board.prior_actions)
        # return calculate_hash(''.join(map(str, board.get_state_complete()))) # slow
    
    def triu2adj(self, board_triu): 
        assert len(board_triu) == self.order*(self.order-1)//2
        adj_matrix = np.zeros((self.order, self.order), dtype=int)
        i = np.triu_indices(self.order, k=1) # k=1 to exclude the diagonal
        col_wise_sort = i[1].argsort()
        i_new = (i[0][col_wise_sort], i[1][col_wise_sort])
        adj_matrix[i_new] = board_triu
        return adj_matrix

    def print_graph(self, adj_matrix):
        G = nx.from_numpy_array(adj_matrix)
        nx.draw(G, with_labels=True, labels={k:k+1 for k in range(self.order)})
        plt.tight_layout()
        plt.savefig("Graph.png", format="PNG")
        wandb.log({"example sat": wandb.Image("Graph.png")})

# from pysat.solvers import Solver
# from pysat.formula import CNF
# from threading import Timer
# import time

# solver_names = ['minisat22', 'glucose4'] # 'cadical153'

# cnf = CNF(from_file='constraints_19_c_100000_2_2_0_final.simp')

# def interrupt(s):
#     s.interrupt()

# for s_name in solver_names:
#     print(s_name)
#     st = time.time()
#     solver = Solver(name=s_name, bootstrap_with=cnf, use_timer=True)
#     print(time.time()-st)
#     # res = solver.solve(assumptions=[45])
#     # t = solver.time()
#     # print(s_name, res, t)
#     # print(solver.accum_stats())
#     # print("------")

#     if s_name not in ['cadical103', 'cadical153', 'lingeling']:
#         try:
#             timer = Timer(50, interrupt, [solver])
#             timer.start()
#             print(solver.solve_limited(expect_interrupt=True))
#             print(solver.time())
#             print(solver.accum_stats())
#             solver.clear_interrupt()
#             print("------")
#         except Exception as e:
#             print(e)
#             print("------")

#     #     try:
#     #         solver.conf_budget(2000)  # getting at most 2000 conflicts
#     #         print(solver.solve_limited())
#     #         print(solver.time())
#     #         print(solver.accum_stats())
#     #         print("------")
#     #     except Exception as e:
#     #         print(e)
#     #         print("------")
    
#     #     try:
#     #         r = solver.propagate(assumptions=[45, 90, 10])
#     #         print(solver.time())
#     #         print(r)
#     #     except Exception as e:
#     #         print(e)

#     print("=======================")