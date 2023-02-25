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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('json_config', type=str, help='Json with experiment design')
    args_cl = parser.parse_args()
    config = json.load(open(args_cl.json_config, 'r'))
    train_path = config['training_dir']
    val_path = config['validation_dir']
    smt_ext = config['file_ext']
    mv_str = config['tactics_config']['all_tactics']
    train_total_timeout = config['train_total_timeout']
    val_total_timeout = config['val_total_timeout']
    train_tactic_timeout = config["train_tactic_timeout"]
    val_tactic_timeout = config["val_tactic_timeout"]
    stats = config['AProVE_min_max'] #change this name: more general
    coach_args = dotdict(config['coach_args'])
    log.info(f'Loading {SMTGame.__name__}...')
    g = SMTGame(benchmarkPath = train_path, ext = smt_ext, moves_str = mv_str, stats = stats, total_timeout = train_total_timeout, tactic_timeout = train_tactic_timeout, train = True)
    g_val = SMTGame(benchmarkPath = val_path, ext = smt_ext, moves_str = mv_str, stats = stats, total_timeout = val_total_timeout, tactic_timeout = val_tactic_timeout, train = False)

    log.info('Loading %s...', snn.__name__)
    nnet = snn(g)

    if coach_args.load_model:
        log.info('Loading checkpoint "%s/%s"...', coach_args.load_folder_file[0], coach_args.load_folder_file[1])
        nnet.load_checkpoint(coach_args.load_folder_file[0], coach_args.load_folder_file[1])
    else:
        log.warning('Not loading a checkpoint!')


    nnet.save_checkpoint(folder=coach_args.checkpoint, filename='best.pth.tar')

    log.info('Loading the Coach...')
    c = Coach(g, g_val, coach_args)

    if coach_args.load_model:
        log.info("Loading 'trainExamples' from file...")
        c.loadTrainExamples()

    log.info('Starting the learning process 🎉')
    c.learn()

    # accRlimit_all = np.array(g.accRlimit_all)
    # print(f"min accRlimit_all: {np.min(accRlimit_all)}, max accRlimit_all: {np.max(accRlimit_all)}, mean accRlimit_all: {np.mean(accRlimit_all)}, std accRlimit_all: {np.std(accRlimit_all)}")

if __name__ == "__main__":
    main()
