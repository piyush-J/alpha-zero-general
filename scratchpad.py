from datetime import datetime
import pickle
import random

MAX_LITERALS_17 = (17*16)//2

MAX_LITERALS_18 = (18*17)//2

literals_pos = list(range(1,MAX_LITERALS_17+1))
literals_neg = [-l for l in literals_pos]
literals_all = literals_pos + literals_neg
vars_all = [MAX_LITERALS_17 + (-c) if c<0 else c for c in literals_all]
lit2var = dict(zip(literals_all, vars_all))
var2lit = dict(zip(vars_all, literals_all))

literals_pos_18 = list(range(1,MAX_LITERALS_18+1))
literals_neg_18 = [-l for l in literals_pos_18]
literals_all_18 = literals_pos_18 + literals_neg_18
vars_all_18 = [MAX_LITERALS_18 + (-c) if c<0 else c for c in literals_all_18]
lit2var_18 = dict(zip(literals_all_18, vars_all_18))
var2lit_18 = dict(zip(vars_all_18, literals_all_18))

with open('test.cubes_mcts_qsa.pkl', 'rb') as f:
    data = pickle.load(f)
    print(data)

for k, v in data.items():
    if k[1] > MAX_LITERALS_17:
        # replace with 18 literals
        new_key = (k[0], lit2var_18[var2lit[k[1]]])
        data[new_key] = v
        del data[k]

print(data)
