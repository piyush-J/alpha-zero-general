import logging

import coloredlogs

from Coach import Coach

from ksgraph.NNet import NNetWrapper as ksnn
from ksgraph.KSGame import KSGame

from utils import *

import wandb

log = logging.getLogger(__name__)

coloredlogs.install(level='INFO')  # Change this to DEBUG to see more info.

args = dotdict({
    'numIters': 10,           # TODO: Change this to 1000
    'numEps': 50,              # Number of complete self-play games to simulate during a new iteration.
    'tempThreshold': 10,        #
    'updateThreshold': 0.6,     # During arena playoff, new neural net will be accepted if threshold or more of games are won.
    'maxlenOfQueue': 200000,    # Number of game examples to train the neural networks. #TODO: Change this to 200000
    'numMCTSSims': 25,          # Number of games moves for MCTS to simulate.
    'arenaCompare': 20,         # Number of games to play during arena play to determine if new net will be accepted.
    'cpuct': 1,                 # controls the amount of exploration

    'checkpoint': './temp/',
    'load_model': False,
    'load_folder_file': ('/dev/models/8x100x50','best.pth.tar'),
    'numItersForTrainExamplesHistory': 20,

    'CCenv': False,
    'model_name': 'DNN',
    'model_notes': 'Basic DNN model during initial testing',
    'model_mode': 'mode-0',
    'phase': 'initial-testing',

    'debugging': False,
})


def main():

    # wandb login

    if args.debugging:
        wandb.init(mode="disabled")
    else:
        wandb.init(reinit=True, 
                    project="AlphaSAT", 
                    tags=[args.model_name, args.model_mode, args.phase], 
                    notes=args.model_notes, 
                    settings=wandb.Settings(start_method='fork' if args.CCenv else 'thread'), 
                    save_code=True)

    wandb.config.update(args)

    log.info(f'Loading {KSGame.__name__}...')
    g = KSGame() 
    log.info('Loading %s...', ksnn.__name__)
    nnet = ksnn(g)

    if args.load_model:
        log.info('Loading checkpoint "%s/%s"...', args.load_folder_file[0], args.load_folder_file[1])
        nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])
    else:
        log.warning('Not loading a checkpoint!')

    log.info('Loading the Coach...')
    c = Coach(g, nnet, args)

    if args.load_model:
        log.info("Loading 'trainExamples' from file...")
        c.loadTrainExamples()

    log.info('Starting the learning process 🎉')
    c.learn()

    # TODO: wandb.save(f"saved_models/{args.model_name}_epc{epoch}_acc{test_acc:.4f}.pt")

if __name__ == "__main__":
    main()
