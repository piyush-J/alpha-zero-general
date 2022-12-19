import logging
import numpy as np
import coloredlogs

from Coach import Coach

# from qzero_planning.NNet import NNetWrapper as pnn
# from qzero_planning.PlanningGame import PlanningGame
# from qzero_planning.PlanningLogic import DomainAction, MinSpanTimeRewardStrategy, RelativeProductRewardStrategy

from smt.NNet import NNetWrapper as snn
from smt.SMTGame import SMTGame

from utils import *

log = logging.getLogger(__name__)

coloredlogs.install(level='INFO')  # Change this to DEBUG to see more info.

args = dotdict({
    'numIters': 1000,           # TODO: Change this to 1000
    'numEps': 216,              # Number of complete self-play games to simulate during a new iteration. # trainset size
    'tempThreshold': 15,        #
    'updateThreshold': 0.6,     # During arena playoff, new neural net will be accepted if threshold or more of games are won.
    'maxlenOfQueue': 200000,    # Number of game examples to train the neural networks. #TODO: Change this to 200000
    'numMCTSSims': 50,          # Number of games moves for MCTS to simulate.
    'arenaCompare': 481,         # Number of games to play during arena play to determine if new net will be accepted.
    'cpuct': 1,                 # controls the amount of exploration

    'checkpoint': './temp/',
    'load_model': False,
    'load_folder_file': ('/dev/models/8x100x50','best.pth.tar'),
    'numItersForTrainExamplesHistory': 20,

})


def main():
    moves_str=("simplify", "smt") # , "bit-blast", "propagate-values", "ctx-simplify", "elim-uncnstr", "solve-eqs", "lia2card",  "max-bv-sharing", "nla2bv", "qfnra-nlsat", "cofactor-term-ite")
    log.info(f'Loading {SMTGame.__name__}...')
    g = SMTGame(benchmarkPath = "smt/example/debug", ext = "smt2", moves_str = moves_str) # "smt/example/qf_nia/AProVE/test"
    g_val = SMTGame(benchmarkPath = "smt/example/debug", ext = "smt2", moves_str = moves_str)

    log.info('Loading %s...', snn.__name__)
    nnet = snn(g)

    if args.load_model:
        log.info('Loading checkpoint "%s/%s"...', args.load_folder_file[0], args.load_folder_file[1])
        nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])
    else:
        log.warning('Not loading a checkpoint!')

    log.info('Loading the Coach...')
    c = Coach(g, g_val, nnet, args)

    if args.load_model:
        log.info("Loading 'trainExamples' from file...")
        c.loadTrainExamples()

    log.info('Starting the learning process 🎉')
    c.learn()

    accRlimit_all = np.array(g.accRlimit_all)
    print(f"min accRlimit_all: {np.min(accRlimit_all)}, max accRlimit_all: {np.max(accRlimit_all)}, mean accRlimit_all: {np.mean(accRlimit_all)}, std accRlimit_all: {np.std(accRlimit_all)}")

if __name__ == "__main__":
    main()
