import logging
import time
import os

import coloredlogs
from matplotlib import pyplot as plt

from Coach import Coach

# from ksgraph.NNet import NNetWrapper as ksnn
from ksgraph.KSGame import KSGame

from utils import *

import wandb

import argparse
import random
import numpy as np

log = logging.getLogger(__name__)

coloredlogs.install(level='INFO')  # Change this to DEBUG to see more info.

random.seed(42)
np.random.seed(42)

args_f = {
    'numIters': 1,           # TODO: Change this to 1000
    'numEps': 1,              # Number of complete self-play games to simulate during a new iteration.
    # 'tempThreshold': 5,        #
    'updateThreshold': None,     # During arena playoff, new neural net will be accepted if threshold or more of games are won.
    'maxlenOfQueue': 200000,    # Number of game examples to train the neural networks.
    # 'numMCTSSims': 300,          # Number of games moves for MCTS to simulate.
    'arenaCompare': 1,         # TODO: change this to 20 or 40 # Number of games to play during arena play to determine if new net will be accepted.
    # 'cpuct': 0.1,                 # controls the amount of exploration; keeping high for MCTSmode 0

    'checkpoint': './temp/',
    'load_model': False,
    'load_folder_file': ('/dev/models/8x100x50','best.pth.tar'),
    'numItersForTrainExamplesHistory': 20,

    'CCenv': True,
    'model_name': 'MCTS',
    'model_notes': 'MCTS without NN',
    'model_mode': 'mode-0',
    'phase': 'scaling',
    'version': 'v1',

    'debugging': True,
    'verbose': True,
    'wandb_logging': False,

    'MCTSmode': 0, # mode 0 - no NN, mode 1 - NN with eval_var (no march call), mode 2 - NN with eval_cls (with march call)
    'nn_iter_threshold': 5, # threshold for the number of iterations after which the NN is used for MCTS

    # 'order': 18,
    # 'MAX_LITERALS': 18*17//2,
    'STATE_SIZE': 10,
    # 'STEP_UPPER_BOUND': 20, # max depth of CnC
    # 'VARS_TO_ELIM': 20, # max number of vars to be eliminated
    'STEP_UPPER_BOUND_MCTS': 20, # max depth of MCTS

    'LIMIT_TOP_3': True
}


def main(args_parsed):
    args = dotdict({**args_f, **vars(args_parsed)})

    if args.prod:
        args['debugging'] = False # force debugging to be False
        args['verbose'] = False # force verbose to be False

    if args.d != -1:
        args['STEP_UPPER_BOUND'] = args_parsed.d
    else:
        args['STEP_UPPER_BOUND'] = None
    
    if args.n != -1:
        args['VARS_TO_ELIM'] = args_parsed.n
    else:
        args['VARS_TO_ELIM'] = None
    
    args['MAX_LITERALS'] = args_parsed.m

    if args.debugging: print(args)

    # wandb login

    if not args.wandb_logging:
        wandb.init(mode="disabled")
    else:
        wandb.init(reinit=True, 
                    name=args.version+"_"+args.model_name+"_"+args.model_mode+"_"+args.phase,
                    project="AlphaSAT", 
                    tags=[args.model_name, args.model_mode, args.phase, args.version], 
                    notes=args.model_notes, 
                    settings=wandb.Settings(start_method='fork' if args.CCenv else 'thread'), 
                    save_code=True)

    wandb.config.update(args)

    # log.info(f'Loading {KSGame.__name__}...')
    g = KSGame(args=args, filename=args.filename) 
    # log.info('Loading %s...', ksnn.__name__)
    # nnet = ksnn(g)
    nnet = None

    # if args.load_model:
    #     log.info('Loading checkpoint "%s/%s"...', args.load_folder_file[0], args.load_folder_file[1])
    #     nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])
    # else:
    #     log.warning('Not loading a checkpoint!')

    if args.debugging: log.info('Loading the Coach...')
    c = Coach(g, nnet, args)

    # if args.load_model:
    #     log.info("Loading 'trainExamples' from file...")
    #     c.loadTrainExamples()

    # log.info('Starting the learning process 🎉')

    if args.MCTSmode == 0:
        c.nolearnMCTS()
        return

    # else ---
    c.learn()

    # wandb.save('arena_cubes.txt')
    wandb.save('tmp.cnf')
    wandb.save('tmp.cubes')
    wandb.save('trainExamples.pkl')

    if args.MCTSmode == 1:

        data = [[x, y] for (x, y) in zip(g.log_giveup_rew, g.log_eval_var)]
        table = wandb.Table(data=data, columns = ["solver_reward (NA)", "eval_var (NA)"])
        wandb.log({"solver_rew vs eval_var (Non-Arena)" : wandb.plot.scatter(table,
                                    "solver_reward (NA)", "eval_var (NA)")})
        
        data = [[x, y] for (x, y) in zip(g.log_giveup_rewA, g.log_eval_varA)]
        table = wandb.Table(data=data, columns = ["solver_reward (A)", "eval_var (A)"])
        wandb.log({"solver_rew vs eval_var (Arena)" : wandb.plot.scatter(table,
                                    "solver_reward (A)", "eval_var (A)")})
        
        # data = [[s] for s in g.log_giveup_rewA]
        # table = wandb.Table(data=data, columns=["solver_reward"])
        fig = plt.figure(figsize =(10, 7))
        plt.boxplot(g.log_giveup_rewA)
        plt.savefig("boxplot1.png")
        wandb.log({"Solver rewards (Arena)": wandb.Image("boxplot1.png")})
        # wandb.log({'Solver rewards (Arena)': wandb.plot.histogram(table, "solver_reward (A)",
        #                         title="Histogram")})
        
        # data = [[s] for s in g.log_eval_varA]
        # table = wandb.Table(data=data, columns=["eval_var"])
        plt.boxplot(g.log_eval_varA)
        plt.savefig("boxplot2.png")
        wandb.log({"Eval Var (Arena)": wandb.Image("boxplot2.png")})
        # wandb.log({'Eval Var (Arena)': wandb.plot.histogram(table, "eval_var (A)",
        #                         title="Histogram")})
        
        if len(g.log_sat_asgn) > 0:
            log_sat_asgn_set = set([frozenset(asgn) for asgn in g.log_sat_asgn])
            data = [[" ".join([str(x) for x in s])] for s in log_sat_asgn_set]
            table =  wandb.Table(data=data, columns=['SAT Assignment'])
            wandb.log({"sat_model": table})

            asgn = data[0][0] # one sat assignment
            asgn = list(map(int, asgn.split(" ")))
            asgn_pos = [a for a in asgn if a > 0]
            triu = [1 if i+1 in asgn_pos else 0 for i in range(g.edge_count)]
            g.print_graph(g.triu2adj(triu))
        
        # TODO: wandb.save(f"saved_models/{args.model_name}_epc{epoch}_acc{test_acc:.4f}.pt")

if __name__ == "__main__":
    # python -u main.py "constraints_17_c_100000_2_2_0_final.simp" -d 1 -m 136 -o "test.cubes" -order 17 -prod
    # python -u main.py "constraints_18_c_100000_2_2_0_final.simp" -order 18 -n 20 -m 153 -o "e4_18_mcts_best_varpen.cubes" -numMCTSSims 300 -cpuct 3 -varpen 0.1 > cubing_outputs/e4_18_mcts_best_varpen.out 2>&1
    # python -u main.py "constraints_19_c_100000_2_2_0_final.simp" -order 19 -n 20 -m 171 -o "e4_19_mcts_best_varpen.cubes" -numMCTSSims 300 -cpuct 0.5 -varpen 0.1 > cubing_outputs/e4_19_mcts_best_varpen.out 2>&1

    start_time_tool = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument("filename", help="filename of the CNF file", type=str)
    parser.add_argument("-order", help="KS order", type=int)
    parser.add_argument("-n", help="cutoff when n variables are eliminated", type=int, default=-1)
    parser.add_argument("-d", help="cutoff when d depth is reached", type=int, default=-1)
    parser.add_argument("-m", help="only top m variables to be considered for cubing", type=int)
    parser.add_argument("-o", help="output file for cubes")
    
    parser.add_argument("-cpuct", help="cpuct for MCTS", type=float, default=10)
    parser.add_argument("-numMCTSSims", help="MCTS sims", type=int, default=10)
    parser.add_argument("-varpen", help="Variance penalty factor", type=float, default=0)
    parser.add_argument("-nMCTSEndOfG", help="MCTS end of game criteria (n cutoff)", type=int, default=-1)

    parser.add_argument("-prod", action="store_true", help="production mode") 

    args_parsed = parser.parse_args()
    if not args_parsed.prod: print(args_parsed)

    args = parser.parse_args() 

    if args_parsed.n == -1 and args_parsed.d == -1:
        print("Either -n or -d must be specified")
        exit()

    # create cubing_outputs folder if it doesn't exist
    if not os.path.exists(f"cubing_outputs/{args.o}"):
        os.makedirs(f"cubing_outputs/{args.o}")

    main(args_parsed)

    time_elapsed_tool = time.time() - start_time_tool
    print("Tool runtime: ", round(time_elapsed_tool, 3))

