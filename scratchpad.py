# # import hashlib

# # def calculate_hash(string):
# #     sha256_hash = hashlib.sha256()
# #     sha256_hash.update(string.encode('utf-8'))
# #     return sha256_hash.hexdigest()

# # string = "Hello, world!"
# # hash_value = calculate_hash(string)
# # print(hash_value)

# import re
# output = """
# c maximum variable to appear in cubes is 6
# c cubes are emitted to x.cubes
# c the DIMACS p-line indicates a CNF of 6 variables and 5 clauses.
# c init_lookahead: longest clause has size 2
# c number of free entry variables = 2
# c number of free variables = 2
# c highest active variable  = 4
# c |----------------------------------------------------------------|
# alphasat: variable: 4, w-left: 0.000000, w-right: 0.000000
# variable: 4, left: 0.000000, right: 0.000000
# variable 4 with score 0.000000
# alphasat: variable: 3, w-left: 0.000000, w-right: 0.000000
# variable: 3, left: 0.000000, right: 0.000000
# variable 3 with score 0.000000
# selected 4 at 1 with diff 0.000000
# c |****************************************************************
# c
# """

# re_out = re.findall(r"alphasat: variable: (\d+), w-left: (\d+.\d+), w-right: (\d+.\d+)", output)
# re_dict = {int(k):(float(v)+float(w)) for k,v,w in re_out}
# print(re_dict)

# import operator
# print(max(re_dict.items(), key=operator.itemgetter(1))[1])

# print(re.findall(r"selected (-?\d+) at \d+ with diff (\d+)", output))

# # o1 = """
# # c
# # c main():: nodeCount: 0
# # c main():: dead ends in main: 0
# # c main():: lookAheadCount: 98
# # c main():: unitResolveCount: 0
# # c time = 1.14 seconds
# # c main():: necessary_assignments: 68
# # c number of cubes 1, including 1 refuted leaf
# # """

# # print("c number of cubes 1, including 1 refuted leaf" in o1)

# import pysat
# import random
# import time

# from pysat.formula import CNF
# from pysat.solvers import Solver

# formula = CNF()
# formula.append([1, 2, 3])
# formula.append([-1, 2, 3])
# formula.append([-4, 2, 3])
# formula.append([-5, 2, 3])
# formula.append([-6, 2, -3])

# solver = Solver(bootstrap_with=formula)
# print(solver.propagate(assumptions=[-5]))

# import time
# import numpy as np
# import matplotlib.pyplot as plt
# import numpy as np
# import wandb

# wandb.init(reinit=True, 
#         name="try",
#         project="AlphaSAT", 
#         settings=wandb.Settings(start_method='thread'), 
#         save_code=True)


# for i in range(10):
#     random_values = np.random.randn(100)
#     random_ints = np.random.randint(10, size=100)

#     data = [[x, y, z, w] for (x, y, z, w) in zip(random_ints, random_values, random_values, random_values)]

#     table = wandb.Table(data=data, columns = ["level", "Qsa", "best_u", "v"])
#     wandb.log({"MCTS Qsa vs tree depth" : wandb.plot.scatter(table,
#                             "level", "Qsa")})
#     wandb.log({"MCTS best_u vs tree depth" : wandb.plot.scatter(table,
#                             "level", "best_u")})
#     wandb.log({"MCTS value vs tree depth" : wandb.plot.scatter(table,
#                             "level", "v")})
    
#     time.sleep(15)


# for i in random_ints:
#     wandb.log({"random_int": i})

# plt.plot([1, 2, 3, 4])
# plt.ylabel("some interesting numbers")
# wandb.log({"chart": plt})

# # Creating dataset
# np.random.seed(10)
# data = np.random.normal(100, 20, 200)
 
# fig = plt.figure(figsize =(10, 7))
# plt.boxplot(data)
# plt.savefig("boxplot.png")
# wandb.log({"example": wandb.Image("boxplot.png")})

# data = [[x, y] for (x, y) in zip(random_ints, random_values)]
# table = wandb.Table(data=data, columns = ["solver_reward (NA)", "eval_var (NA)"])
# wandb.log({'Solver rewards (Arena)': wandb.plot.histogram(table, "solver_reward (NA)",
#                         title="Histogram")})