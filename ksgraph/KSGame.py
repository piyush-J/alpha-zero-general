import copy
from Game import Game
from .KSLogic import Board, MAX_LITERALS, MAX_CLAUSE_EMBED

import numpy as np
from pysat.formula import CNF

class KSGame(Game):
    def __init__(self, filename="constraints_5"): 
        super(KSGame, self).__init__()
        self.cnf = CNF(from_file=filename)
        order = int(filename.split("_")[-1])
        print("Solving the KS system of order ", order, " with ", len(self.cnf.clauses), " constraints")

        self.edge_dict = {}
        # self.tri_dict = {}
        count = 0
        for j in range(1, order+1):             #generating the edge variables
            for i in range(1, order+1):
                if i < j:
                    count += 1
                    self.edge_dict[(i,j)] = count

        assert MAX_LITERALS >= count # sanity check so that we can encode the edge variables in the action space

        # for a in range(1, order-1):             #generating the triangle variables
        #     for b in range(a+1, order):
        #         for c in range(b+1, order+1):
        #             count += 1
        #             self.tri_dict[(a,b,c)] = count

    def _make_representation(self):
        return Board(cnf=self.cnf, edge_dict=self.edge_dict)

    def get_copy(self):
        return copy.deepcopy(self)

    def getInitBoard(self):
        bd = self._make_representation()
        return bd

    def getEmbedding(self, board):
        return board.get_state()

    def getBoardSize(self):
        return MAX_CLAUSE_EMBED

    def getActionSize(self):
        return MAX_LITERALS*2 + 1 # [e.g., dummy_0, 1 to 10, -1 to -10]
    
    def getNv(self):
        return self.cnf.nv

    def getNextState(self, board, action):
        new_board = board.execute_move(action)
        return new_board

    def getValidMoves(self, board):
        valids = [0]*self.getActionSize()
        legalMoves =  [board.lits2var[l] for l in board.get_legal_moves()]
        for x in legalMoves:
            valids[x]=1
        return np.array(valids)

    def getGameEnded(self, board):
        """
        Input:
            board: current board

        Returns:
            r: 0 if game has not ended, reward otherwise. 
               
        """
        return board.compute_reward() if board.is_done() else None

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
        return ''.join(map(str, board.get_state_complete()))