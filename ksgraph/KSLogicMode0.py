import copy
import itertools
import logging
import operator
import re
import subprocess
from abc import abstractmethod
from dataclasses import dataclass

import numpy as np
from pysat.solvers import Solver
from pysat.formula import CNF

import wandb

from .KSLogic import Board

log = logging.getLogger(__name__)
cnf_obj = None

class BoardMode0(Board):

    def __init__(self, args, cnf, edge_dict, max_metric_val):
        Board.__init__(self, args, cnf, edge_dict)
        self.order = args.order
        self.valid_literals = None
        self.prob = None
        self.march_pos_lit_score_dict = None
        self.current_metric_val = None
        self.max_metric_val = max_metric_val # maximum possible value of the metric (unweighted)

        global cnf_obj
        cnf_obj = cnf

    def __str__(self):
        return f"Board- res: {self.res}, step: {self.step}, total_rew: {self.total_rew:.3f}, prior_actions: {self.prior_actions}"


    def is_giveup(self): # give up if we have reached the upper bound on the number of steps or if there are only 0 or extra lits left
        return self.res is None and (self.step >= self.args.STEP_UPPER_BOUND or len(self.get_legal_literals()) == 0)
    
    def calculate_march_metrics(self):
        # TODO: file saving might cause issue with code parallelization
        if self.args.debugging: log.info(f"Calculating march metrics")
        filename = "tmp.cnf"
        CNF(from_clauses=self.cnf()).to_file(filename)
        if self.args.debugging: log.info(f"Saved to file")
        edge_vars = self.order*(self.order-1)//2 

        # if len(self.prior_actions) > 0:
        #     assert self.cnf_clauses_org + self.prior_actions == self.cnf.clauses # sanity check

        # ../PhysicsCheck/gen_cubes/march_cu/march_cu tmp.cnf -o tmp.cubes -d 1 -m ...
        result = subprocess.run(['../PhysicsCheck/gen_cubes/march_cu/march_cu', 
                                filename,
                                '-o',
                                'tmp.cubes', 
                                '-d', '1', '-m', str(edge_vars)], capture_output=True, text=True)
        output = result.stdout

        # two groups enclosed in separate ( and ) bracket
        # this score considers the product of the two sides which can create problem (Refer Debugging Notes #7)
        # march_pos_lit_score_dict = dict(re.findall(r"alphasat: variable (\d+) with score (\d+)", output))
        # march_pos_lit_score_dict = {int(k):float(v) for k,v in march_pos_lit_score_dict.items()}

        re_out = re.findall(r"alphasat: variable: (\d+), w-left: (\d+.\d+), w-right: (\d+.\d+)", output)
        march_pos_lit_score_dict = {int(k):(float(v)+float(w))/2.0 for k,v,w in re_out} # average of the two sides

        if len(march_pos_lit_score_dict) == 0:
            unsat_check = re.findall(r"c number of cubes (\d+), including (\d+) refuted leaf", output)
            if len(unsat_check) > 0 and unsat_check[0][0] == unsat_check[0][1] in output:
                assert len(unsat_check) == 1
                self.res = 0
            elif "SATISFIABLE" in output:
                self.res = 1
                print("Found SAT!")
                print(output)
                print("Exiting...")
                exit(0)
            elif "s UNKNOWN" in output:
                self.res = 2
            else:
                print("Unknown result with empty dict! Check tmp.cnf and tmp.cubes files")
                print(output)
                exit(0)

        valid_pos_literals = list(march_pos_lit_score_dict.keys())
        valid_neg_literals = [-l for l in valid_pos_literals]

        prob = [0.0 for _ in range(edge_vars*2+1)]
        for l in valid_pos_literals:
            prob[l] = march_pos_lit_score_dict[l]
            prob[self.lits2var[-l]] = march_pos_lit_score_dict[l]
        
        if sum(prob) == 0:
            # uniform distribution
            prob = [1/(edge_vars*2) for _ in range(edge_vars*2+1)]
            prob[0] = 0.0 # 0 is not a valid literal
        else:
            prob = [p/sum(prob) for p in prob] # only for +ve literals

        # normalize the values of the march_pos_lit_score_dict
        for k in march_pos_lit_score_dict.keys():
            march_pos_lit_score_dict[k] /= self.max_metric_val

        max_val = max(march_pos_lit_score_dict.values()) if len(march_pos_lit_score_dict) > 0 else 0
        wandb.log({"depth": self.step, "max_val": max_val})

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
        if self.args.debugging: log.info(f"Executing action {action}")
        new_state = copy.deepcopy(self)
        if self.args.debugging: log.info(f"Deepcopy done")
        new_state.valid_literals = None
        new_state.prob = None
        new_state.march_pos_lit_score_dict = None
        new_state.current_metric_val = None

        new_state.step += 1
        chosen_literal = [new_state.var2lits[action]]
        new_state.prior_actions.append(chosen_literal)
        # new_state.cnf.append(chosen_literal) # append to the cnf object
        # collecting from the parent node's dict; TODO: not considering direction, so choosing the +ve one (abs)
        assert self.march_pos_lit_score_dict is not None
        self.current_metric_val = self.march_pos_lit_score_dict[abs(chosen_literal[0])]
        new_state.total_rew += self.current_metric_val # adding so that the leaves denote the total reward of the path
        new_state.calculate_march_metrics()
        if self.args.debugging: log.info(f"Calculated march metrics")
        return new_state

    def compute_reward(self, eval_cls=False):
        norm_rew = None
        if self.is_done():
            if self.is_win():
                print("Found SAT!")
                print(self.prior_actions)
                print("Exiting...")
                exit(0)
                # return self.total_rew + self.args.STEP_UPPER_BOUND
            elif self.is_fail():
                norm_rew = 1
            elif self.is_unknown(): # results in unknown using march_cu so heavily penalize and don't go down this path
                norm_rew = -1
            elif self.is_giveup(): 
                norm_rew = self.total_rew # if any changes are made here, make sure to change the reward in MCTS.py as well
            else:
                raise Exception("Unknown game state")
            
            if eval_cls:
                return norm_rew * self.max_metric_val
            else:
                wandb.log({"depth": self.step, "norm_rew": norm_rew})
                return norm_rew
        else:
            return None

    @DeprecationWarning
    def eval_var(self): # slow
        march_pos_lit_score_dict = {}
        
        for chosen_literal in list(set(self.get_flattened_clause()) - set([0]+self.extra_lits)):
            print(chosen_literal)
            chosen_literal = int(chosen_literal)
            if abs(chosen_literal) in march_pos_lit_score_dict:
                continue

            with Solver(bootstrap_with=self.cnf()) as solver:
                out = solver.propagate(assumptions=[chosen_literal])
                assert out is not None
                not_unsat, asgn = out

                if not not_unsat: # unsat
                    rew1 = 0
                else:
                    rew1 = len(asgn)

                out = solver.propagate(assumptions=[-chosen_literal])
                assert out is not None
                not_unsat, asgn = out

                if not not_unsat: # unsat
                    rew2 = 0
                else:
                    rew2 = len(asgn)

                if not (rew1 == 0 and rew2 == 0):
                    # one of them is not unsat
                    march_pos_lit_score_dict[abs(chosen_literal)] = rew1 + rew2