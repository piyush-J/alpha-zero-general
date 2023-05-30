import itertools
import numpy as np
from dataclasses import dataclass
from abc import abstractmethod
from .KSLogic import Board, MAX_LITERALS, MAX_CLAUSE_EMBED, STEP_UPPER_BOUND

import copy

from pysat.solvers import Solver
import wandb

class BoardMode0(Board):

    def __init__(self, cnf, edge_dict):
        Board.__init__(self, cnf, edge_dict)

    def is_giveup(self): # give up if we have reached the upper bound on the number of steps or if there are only 0 or extra lits left
        return self.res is None and (self.step > STEP_UPPER_BOUND or len(self.get_legal_literals()) == 0)
    
    def is_win(self):
        return self.res == 1
    
    def is_fail(self):
        return self.res == 0

    def is_done(self):
        return self.is_giveup() or self.is_win() or self.is_fail()
    
    def get_legal_literals(self):
        return set() # TODO
        
    def execute_move(self, action):
        assert self.is_done() == False
        # if action not in [self.lits2var[l] for l in self.get_legal_literals()]:
        #     print("Illegal move!")
        new_state = copy.deepcopy(self)

        new_state.step += 1
        chosen_literal = [new_state.var2lits[action]]
        new_state.prior_actions.append(chosen_literal)
        new_state.total_rew = self.march_metrics(self.cnf_clauses) #TODO: think - rewards keep getting added?

        return new_state

    def compute_reward(self):
        if self.is_done():
            if self.is_win():
                raise ValueError("self.res incorrectly set to 1")
            elif self.is_fail():
                raise ValueError("self.res incorrectly set to 0")
            elif self.is_giveup(): 
                self.cnf_clauses
                #TODO
                
            else:
                raise Exception("Unknown game state")
        else:
            return None
