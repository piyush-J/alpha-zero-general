import copy
import itertools
import re
import subprocess
from abc import abstractmethod
from dataclasses import dataclass

import numpy as np
from pysat.solvers import Solver

import wandb

from .KSLogic import MAX_CLAUSE_EMBED, MAX_LITERALS, STEP_UPPER_BOUND, Board


class BoardMode0(Board):

    def __init__(self, cnf, edge_dict, order):
        Board.__init__(self, cnf, edge_dict)
        self.order = order
        self.valid_literals = None
        self.prob = None
        self.march_pos_lit_score_dict = None
        self.current_metric_val = None

    def is_giveup(self): # give up if we have reached the upper bound on the number of steps or if there are only 0 or extra lits left
        return self.res is None and (self.step > STEP_UPPER_BOUND or len(self.get_legal_literals()) == 0)
    
    def calculate_march_metrics(self):
        # TODO: file saving might cause issue with code parallelization
        filename = "tmp.cnf"
        self.cnf.to_file(filename)
        edge_vars = self.order*(self.order-1)//2 

        result = subprocess.run(['../PhysicsCheck/gen_cubes/march_cu/march_cu', 
                                filename,
                                '-o',
                                'mcts_logs/tmp.cubes', 
                                '-d', '1', '-m', str(edge_vars)], capture_output=True, text=True)
        output = result.stdout

        # two groups enclosed in separate ( and ) bracket
        march_pos_lit_score_dict = dict(re.findall(r"alphasat: variable (\d+) with score (\d+)", output))
        march_pos_lit_score_dict = {int(k):int(v) for k,v in march_pos_lit_score_dict.items()}

        # [best literal with sign, node, diff of the selected literal]
        try:
            march_var_node_score_list = list(map(int, re.findall(r"selected (-?\d+) at (\d+) with diff (\d+)", output)[0]))
        except IndexError:
            print("No literals to choose from!")
            print(output)
            exit(0)

        valid_pos_literals = list(march_pos_lit_score_dict.keys())
        valid_neg_literals = [-l for l in valid_pos_literals]

        prob = [0 for _ in range(edge_vars*2+1)]
        for l in valid_pos_literals:
            prob[l] = march_pos_lit_score_dict[l]
            prob[-l] = march_pos_lit_score_dict[l]
        
        try:
            prob = [p/sum(prob) for p in prob] # only for +ve literals
        except ZeroDivisionError:
            # uniform distribution
            prob = [1/(edge_vars*2) for _ in range(edge_vars*2+1)]

        self.valid_literals = valid_pos_literals + valid_neg_literals # both +ve and -ve literals
        self.prob = prob
        self.march_pos_lit_score_dict = march_pos_lit_score_dict
    
    def get_legal_literals(self):
        assert self.valid_literals is not None
        return set(self.valid_literals)
        
    def execute_move(self, action):
        assert self.is_done() == False
        # if action not in [self.lits2var[l] for l in self.get_legal_literals()]:
        #     print("Illegal move!")
        new_state = copy.deepcopy(self)
        new_state.valid_literals = None
        new_state.prob = None
        new_state.march_pos_lit_score_dict = None
        new_state.current_metric_val = None

        new_state.step += 1
        chosen_literal = [new_state.var2lits[action]]
        new_state.prior_actions.append(chosen_literal)
        new_state.cnf.append(chosen_literal) # append to the cnf object
        # collecting from the parent node's dict; TODO: not considering direction, so choosing the +ve one (abs)
        assert self.march_pos_lit_score_dict is not None
        self.current_metric_val = self.march_pos_lit_score_dict[abs(chosen_literal[0])]
        new_state.total_rew += self.current_metric_val # adding so that the leaves denote the total reward of the path
        new_state.calculate_march_metrics()
        return new_state

    def compute_reward(self):
        if self.is_done():
            if self.is_win():
                raise ValueError("self.res incorrectly set to 1")
            elif self.is_fail():
                raise ValueError("self.res incorrectly set to 0")
            elif self.is_giveup(): 
                return self.total_rew
            else:
                raise Exception("Unknown game state")
        else:
            return None
