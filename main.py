import argparse
import json
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

import functools
print = functools.partial(print, flush=True)

log = logging.getLogger(__name__)

coloredlogs.install(level='INFO')  # Change this to DEBUG to see more info.

# args = dotdict({
#     'numIters': 3,
#     'numEps': 52,              # Number of complete self-play games to simulate during a new iteration. # with the current setting it can only be training set size #TO_DO: do not hard code it
#     'tempThreshold': 15,        #
#     'updateThreshold': 0.6,     # During arena playoff, new neural net will be accepted if threshold or more of games are won.
#     'maxlenOfQueue': 200000,    # Number of game examples to train the neural networks. #TODO: Change this to 200000
#     'numMCTSSims': 25,          # Number of games moves for MCTS to simulate.
#     'arenaCompare': 30,         # Number of games to play during arena play to determine if new net will be accepted. # now can only be validation set size #TO_DO: do not hard code it
#     'cpuct': 1,                 # controls the amount of exploration
#
#     'checkpoint': './temp/',
#     'load_model': False,
#     'load_folder_file': ('./temp/','best.pth.tar'),
#     'numItersForTrainExamplesHistory': 20,
#
# })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('json_config', type=str, help='Json with experiment design')
    args_cl = parser.parse_args()
    config = json.load(open(args_cl.json_config, 'r'))
    train_path = config['training_dir']
    val_path = config['validation_dir']
    smt_ext = config['file_ext']
    mv_str = config['tactics_config']['all_tactics']
    stats = config['AProVE_min_max']
    coach_args = dotdict(config['coach_args'])
    log.info(f'Loading {SMTGame.__name__}...')
    g = SMTGame(benchmarkPath = train_path, ext = smt_ext, moves_str = mv_str, stats = stats)
    g_val = SMTGame(benchmarkPath = val_path, ext = smt_ext, moves_str = mv_str, stats = stats)

    log.info('Loading %s...', snn.__name__)
    nnet = snn(g)

    if coach_args.load_model:
        log.info('Loading checkpoint "%s/%s"...', coach_args.load_folder_file[0], coach_args.load_folder_file[1])
        nnet.load_checkpoint(coach_args.load_folder_file[0], coach_args.load_folder_file[1])
    else:
        log.warning('Not loading a checkpoint!')

    log.info('Loading the Coach...')
    c = Coach(g, g_val, nnet, coach_args)

    if coach_args.load_model:
        log.info("Loading 'trainExamples' from file...")
        c.loadTrainExamples()

    log.info('Starting the learning process 🎉')
    c.learn()

    accRlimit_all = np.array(g.accRlimit_all)
    print(f"min accRlimit_all: {np.min(accRlimit_all)}, max accRlimit_all: {np.max(accRlimit_all)}, mean accRlimit_all: {np.mean(accRlimit_all)}, std accRlimit_all: {np.std(accRlimit_all)}")

if __name__ == "__main__":
    main()
