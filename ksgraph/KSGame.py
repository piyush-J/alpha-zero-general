import copy
from Game import Game
from .KSLogic import Board
import numpy as np
import networkx as nx

from pysat.formula import CNF
from pysat.solvers import Solver

ORDER = 4 # order of the KS system

class KSGame(Game):
    def __init__(self, n=1): 
        super(KSGame, self).__init__()
        self.n = n # size of the adjacency matrix
        self.cnf = CNF(from_file="constraints_5")
        self.solver = Solver(bootstrap_with=self.cnf)

        self.edge_dict = {}
        self.tri_dict = {}
        count = 0
        for j in range(1, ORDER+1):             #generating the edge variables
            for i in range(1, ORDER+1):
                if i < j:
                    count += 1
                    self.edge_dict[(i,j)] = count
        # for a in range(1, ORDER-1):             #generating the triangle variables
        #     for b in range(a+1, ORDER):
        #         for c in range(b+1, ORDER+1):
        #             count += 1
        #             self.tri_dict[(a,b,c)] = count

    def _make_representation(self):
        return Board(n=self.n, solver=self.solver, edge_dict=self.edge_dict)

    def get_copy(self):
        return copy.deepcopy(self)

    def getInitBoard(self):
        """
        Returns:
            startBoard: a representation of the board (ideally this is the form
                        that will be the input to your neural network)
        """
        r = self._make_representation()
        return r.triu # np.array([0])

    def getBoardSize(self):
        return self.n*(self.n-1)//2 # upper triangular elements

    def getActionSize(self):
        """
        Returns:
            actionSize: number of all possible actions
        """
        return 2**ORDER # TODO: later shift to self.n + 1

    def getNextState(self, board, action):
        """
        Input:
            board: current board (np.array)
            action: action taken by current player (int)

        Returns:
            nextBoard: board after applying action
        """
        r = self._make_representation()
        r.triu = np.copy(board)
        r.execute_move(action)
        return r.triu

    def getValidMoves(self, board):
        """
        Input:
            board: current board

        Returns:
            validMoves: a binary vector of length self.getActionSize(), 1 for
                        moves that are valid from the current board,
                        0 for invalid moves
        """
        # return a fixed size binary vector
        valids = [0]*self.getActionSize()
        r = self._make_representation()
        r.triu = np.copy(board)
        legalMoves =  r.get_legal_moves()
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
        r = self._make_representation()
        r.triu = np.copy(board)
        return r.compute_reward() if r.is_done() else None

    # convert the 2-D adjacency matrix to a 1-D vector of the upper triangular elements
    def adj2triu(self, adj_matrix): 
        assert len(adj_matrix) == self.n
        i = np.triu_indices(self.n, k=1) # k=1 to exclude the diagonal
        col_wise_sort = i[1].argsort()
        i_new = (i[0][col_wise_sort], i[1][col_wise_sort])
        board_triu = adj_matrix[i_new]
        return board_triu

    # convert the a 1-D vector of the upper triangular elements to a 2-D adjacency matrix
    def triu2adj(self, board_triu): 
        assert len(board_triu) == self.n*(self.n-1)//2
        adj_matrix = np.zeros((self.n, self.n), dtype=int)
        i = np.triu_indices(self.n, k=1) # k=1 to exclude the diagonal
        col_wise_sort = i[1].argsort()
        i_new = (i[0][col_wise_sort], i[1][col_wise_sort])
        adj_matrix[i_new] = board_triu
        return adj_matrix

    # create a isomorphic graph from the adjacency matrix
    def create_graph(self, adj_matrix, permutation_matrix):
        G = nx.from_numpy_array(adj_matrix)
        H = nx.relabel_nodes(G, dict(zip(G.nodes(), permutation_matrix)))
        return H

    # print networkx graph from adjacency matrix
    def print_graph(self, adj_matrix):
        G = nx.from_numpy_array(adj_matrix)
        nx.draw(G, with_labels=True)

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
        return ''.join(map(str, board))