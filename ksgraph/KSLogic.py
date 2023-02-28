import itertools
import numpy as np
from dataclasses import dataclass
from abc import abstractmethod

import copy

from pysat.solvers import Solver

MAX_LITERALS = 10
MAX_CLAUSE_EMBED = 500
STEP_UPPER_BOUND = MAX_LITERALS//2

class Board:

    def __init__(self, cnf, edge_dict):
        self.cnf_clauses = copy.deepcopy(cnf.clauses) # first call would have CNF object
        self.nlits = cnf.nv # number of variables in the CNF formula
        self.extra_lits = list(range(MAX_LITERALS+1, self.nlits+1, 1))+list(range(-MAX_LITERALS-1, -self.nlits-1, -1)) # extra variabls not part of the action space
        self.res = None
        self.edge_dict = edge_dict

        self.step = 0
        self.total_rew = 0
        self.sat_or_unsat_leaf = 0
        self.prior_actions = []
        self.sat_unsat_actions = [] # actions that lead to a sat or unsat leaf - to be removed from the legal action space

        literals_pos = list(range(1,MAX_LITERALS+1))
        literals_neg = [-l for l in literals_pos]
        literals_all = literals_pos + literals_neg
        vars_all = [MAX_LITERALS + (-c) if c<0 else c for c in literals_all]
        self.lits2var = dict(zip(literals_all, vars_all))
        self.var2lits = dict(zip(vars_all, literals_all))

    def __str__(self):
        return f"nlits: {self.nlits}, res: {self.res}, step: {self.step}, total_rew: {self.total_rew}, sat_or_unsat_leaf: {self.sat_or_unsat_leaf}, prior_actions: {self.prior_actions}, sat_unsat_actions: {self.sat_unsat_actions}"

    def is_giveup(self):
        return self.step > STEP_UPPER_BOUND

    def is_win(self):
        return self.res == 1
    
    def is_fail(self):
        return self.res == 0

    def is_done(self):
        return self.is_giveup() or self.is_win() or self.is_fail()
        
    def get_state(self):
        clauses = copy.deepcopy(self.cnf_clauses)
        for i, c in enumerate(clauses):
            clauses[i].append(0) # add the clause separator
        
        clauses = list(itertools.chain.from_iterable(clauses)) # flatten the list
        clauses = [self.nlits + (-c) if c<0 else c for c in clauses] # treat the negative literals as a new literal (for convenient MCTS action space)
        clauses_padded = clauses[:MAX_CLAUSE_EMBED] + [0]*(MAX_CLAUSE_EMBED-len(clauses)) # pad the list with 0s
        return np.array(clauses_padded)

    def get_state_complete(self): # no truncation or padding
        clauses = copy.deepcopy(self.cnf_clauses)
        for i, c in enumerate(clauses):
            clauses[i].append(0) # add the clause separator
        
        clauses = list(itertools.chain.from_iterable(clauses)) # flatten the list
        clauses = [self.nlits + (-c) if c<0 else c for c in clauses] # treat the negative literals as a new literal (for convenient MCTS action space)
        return np.array(clauses)
    
    def get_flattened_clause(self): # no mapping or truncation or padding
        clauses = copy.deepcopy(self.cnf_clauses)
        for i, c in enumerate(clauses):
            clauses[i].append(0) # add the clause separator
        
        clauses = list(itertools.chain.from_iterable(clauses)) # flatten the list
        return np.array(clauses)

    def get_legal_moves(self):
        if self.is_done():
            raise Exception("Game is already over")
        else:
            return set(self.get_flattened_clause()) - set([0]+[self.var2lits[v] for v in self.sat_unsat_actions]+self.extra_lits) # remove the clause separator from the list of legal moves
        
    def execute_move(self, action):
        assert self.is_done() == False
        new_state = copy.deepcopy(self)

        new_state.step += 1
        chosen_literal = [new_state.var2lits[action]]
        self.prior_actions.append(new_state.var2lits[action])
        
        with Solver(bootstrap_with=new_state.cnf_clauses) as solver:
            out = solver.propagate(assumptions=chosen_literal)
            assert out is not None
            not_unsat, asgn = out
        
        new_state.total_rew += len(asgn) # reward is the number of literals that are assigned (eval_var)

        if not not_unsat: # unsat
            new_state.res = 0
            new_state.cnf_clauses = [[]]
            new_state.sat_or_unsat_leaf += 1
            self.sat_or_unsat_leaf += 1 # update the parent too, so that you can propagate to the cutoff leaf
            self.sat_unsat_actions.append(action) # add the action to the list of actions (of the parent) that lead to a sat or unsat leaf

        else:
            clauses_interm = [c for c in new_state.cnf_clauses if all(r not in c for r in chosen_literal)] # remove the clauses that contain the chosen literal
            new_state.cnf_clauses = [[l for l in c if all(l!=-r for r in chosen_literal)] for c in clauses_interm] # remove the negation of chosen literal from the remaining clauses
            if new_state.cnf_clauses == []: # sat
                new_state.res = 1
                new_state.sat_or_unsat_leaf += 1
                self.sat_or_unsat_leaf += 1 # update the parent too, so that you can propagate to the cutoff leaf
                self.sat_unsat_actions.append(action) # add the action to the list of actions (of the parent) that lead to a sat or unsat leaf

        return new_state

    def compute_reward(self):
        if self.is_done():
            if self.is_win() or self.is_fail():
                return -1 # dicourage the agent from exploring this again
            elif self.is_giveup(): # call the solver to get the result
                with Solver(bootstrap_with=self.cnf_clauses, use_timer=True) as solver:
                    res = solver.solve()
                    time_s = solver.time()
                    assert time_s is not None
                    return -time_s/10 # penalty is the time it takes to solve the problem (seconds / 10)
            else:
                raise Exception("Unknown game state")
        else:
            return None
