import numpy as np
from dataclasses import dataclass
from abc import abstractmethod
from .KSGame import ORDER

class Board:

    def __init__(self, n, solver, edge_dict):
        self.n = n
        self.triu = np.zeros((n))
        self.solver = solver
        self.edge_dict = edge_dict

        i = np.triu_indices(self.n, k=1) # k=1 to exclude the diagonal
        col_wise_sort = i[1].argsort()
        self.i_new = (i[0][col_wise_sort], i[1][col_wise_sort]) # indices of the upper triangular elements in a column-wise sorted matrix

    def _action_to_move(self, action: int): # dec to bin
        return f'{action:#0{ORDER-1+2}b}'[2:] # -1 because the diagonals are 0 by default, +2 for 0b

    def _move_to_action(self, move: str): # bin to dec
        return int(move, 2)

    def is_done(self):
        return self.n == ORDER-1 # For a 4x4 matrix, the last column is filled without the diagonal (=3)

    def get_legal_moves(self) -> set[int]: # TODO: retrieve from dict and remove the invalid moves
        if not self.is_done():
            return set(range(2**self.n)) # e.g., 0 or 1 for n=1 (filling the first row, second column upper traingle)
        else:
            return set()

    def execute_move(self, action):
        bin_str = self._action_to_move(action)
        triu_str = ''.join(map(str, self.triu))
        triu_str = triu_str + bin_str
        self.triu = np.array(list(map(int, triu_str)))
        # TODO: call the CAS implementation and update the valid actions list for the parent node vs get symmetry call during executeEpisode
        # TODO: do you need something like self.legal_actions.remove(self._move_to_action(bin_str)) ?

    def compute_reward(self):
        if self.is_done():
            triu = self.triu.astype(bool)
            assumptions = [self.edge_dict[(r,c)] for r,c in zip(self.i_new[0][triu]+1, self.i_new[1][triu]+1)] # +1 because the indices start from 1
            res = self.solver.solve(assumptions=assumptions)
            if res:
                return 1
            else:
                # self.solver.get_core() # TODO: unsat core to be always blocked
                return -1
        else:
            return None
