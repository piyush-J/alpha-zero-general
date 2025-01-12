import Arena
from Coach import Coach
from MCTS import MCTS
from cdrl.CdrlGame import CdrlGame
from cdrl.pytorch.NNet import NNetWrapper as NNet

import numpy as np
from utils import *

# Check output of the random model and the trained model

g = CdrlGame()
args1 = dotdict({'numMCTSSims': 50, 'cpuct':1.0, 'arenaCompare': 1})

n1 = NNet(g)

print("Random model")
# random initialization of MCTS and MCTS with neural network
mcts = MCTS(g, n1, args1)
c = Coach(g, n1, args1)
c.evaluate_model(mcts)

print("Trained model")
# play one round of game with trained model
n1.load_checkpoint('./temp','best.pth.tar')
mcts = MCTS(g, n1, args1)
c = Coach(g, n1, args1)
c.evaluate_model(mcts)

