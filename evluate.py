import glob
import csv

from Runner import Runner
from smt.SMTLogic import Board
from smt.NNet import NNetWrapper as snn
from smt.SMTGame import SMTGame

# change this to config
moves_str = ["simplify","smt","bit-blast","propagate-values","ctx-simplify","elim-uncnstr","solve-eqs","lia2card","max-bv-sharing","nla2bv","qfnra-nlsat","cofactor-term-ite","sat"]

stats = {
    "arith-max-deg": [0, 15],
    "arith-avg-deg": [0, 3],
    "arith-max-bw": [0, 2],
    "arith-avg-bw": [1, 2],
    "memory": [15, 20],
    "size": [4, 6074],
    "num-exprs": [10, 60000],
    "num-consts": [0, 1000],
    "num-arith-consts": [0, 1000]
}

g = SMTGame(benchmarkPath = "smt/example/qf_nia/john2/test/", ext = "smt2", moves_str = moves_str, stats = stats, tactic_timeout = 60)

nnet = snn(g)
nnet.load_checkpoint("./temp/", "best.pth.tar")

fLst = []
for fm in sorted(glob.glob("smt/example/qf_nia/john2/test/*.smt2")):
    fLst.append(fm)

time_out = 180 #seconds
# threads = []

with open("policy_eval_Feb9_2.csv", 'w') as f:
    counter = 0
    header = ['formula_id', 'res', 'rlimit', 'runtime', 'nn_time', 'solver_time']
    writer = csv.writer(f)
    writer.writerow(header)
    for fPath in fLst:
        # stratgies: "smt" "(then simplify smt)"
        if counter > 106:
            bd = Board(counter, fPath, moves_str, None, stats, False)
            thread = Runner(nnet, bd, 180, 60, True, "log_file_0209.txt")
            thread.start() #not pararell now
            thread.join(180)
            res, rlimit, runtime, nn_time, solver_time = thread.collect()
            r = [counter, res, rlimit, runtime, nn_time, solver_time]
            print(r)
            writer.writerow(r)
            f.flush()
            counter += 1
