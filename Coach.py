import logging
import os
import sys
from collections import deque
from pickle import Pickler, Unpickler
from random import shuffle

import datetime
import numpy as np
from tqdm import tqdm

from Arena import PlanningArena
from MCTS import MCTS

log = logging.getLogger(__name__)

import functools
print = functools.partial(print, flush=True)


class Coach():
    """
    This class executes the self-play + learning. It uses the functions defined
    in Game and NeuralNet. args are specified in main.py.
    """

    def __init__(self, game, game_validation, nnet, args):
        self.game = game
        self.game_validation = game_validation
        self.nnet = nnet
        self.pnet = self.nnet.__class__(self.game)  # the competitor network
        self.args = args
        self.log_to_file = self.args.log_to_file
        self.filename = "out-{date:%Y-%m-%d_%H-%M-%S}.log".format(date=datetime.datetime.now())
        self.mcts = MCTS(self.nnet, self.args, self.filename)
        self.trainExamplesHistory = []  # history of examples from args.numItersForTrainExamplesHistory latest iterations
        self.skipFirstSelfPlay = False  # can be overriden in loadTrainExamples()


    # def executeEpisode(self):
    #     """
    #     This function executes one episode of self-play, starting with player 1.
    #     As the game is played, each turn is added as a training example to
    #     trainExamples. The game is played till the game ends. After the game
    #     ends, the outcome of the game is used to assign values to each example
    #     in trainExamples.

    #     It uses a temp=1 if episodeStep < tempThreshold, and thereafter
    #     uses temp=0.

    #     Returns:
    #         trainExamples: a list of examples of the form (canonicalBoard, currPlayer, pi,v)
    #                        pi is the MCTS informed policy vector, v is +1 if
    #                        the player eventually won the game, else -1.
    #     """
    #     trainExamples = []
    #     board = self.game.getInitBoard()
    #     self.curPlayer = 1
    #     episodeStep = 0

    #     while True:
    #         episodeStep += 1
    #         canonicalBoard = self.game.getCanonicalForm(board, self.curPlayer)
    #         temp = int(episodeStep < self.args.tempThreshold)

    #         pi = self.mcts.getActionProb(canonicalBoard, temp=temp)
    #         sym = self.game.getSymmetries(canonicalBoard, pi)
    #         for b, p in sym:
    #             trainExamples.append([b, self.curPlayer, p, None])

    #         action = np.random.choice(len(pi), p=pi)
    #         board, self.curPlayer = self.game.getNextState(board, self.curPlayer, action)

    #         r = self.game.getGameEnded(board, self.curPlayer)

    #         if r != 0:
    #             return [(x[0], x[2], r * ((-1) ** (x[1] != self.curPlayer))) for x in trainExamples]

    def executeEpisode(self):
        """
        This function executes one episode of self-play.
        As the game is played, each turn is added as a training example to
        trainExamples. The game is played till the game ends. After the game
        ends, the outcome of the game is used to assign values to each example
        in trainExamples.

        It uses a temp=1 if episodeStep < tempThreshold, and thereafter
        uses temp=0.

        Returns:
            trainExamples: a list of examples of the form (canonicalBoard, currPlayer, pi,v)
                           pi is the MCTS informed policy vector, v is +1 if
                           the player eventually won the game, else -1.
        """
        trainExamples = []
        game = self.game.get_copy()
        # board = self.game.getInitBoard()
        board = game.getInitBoard()
        episodeStep = 0

        while True:
            episodeStep += 1
            temp = int(episodeStep < self.args.tempThreshold)

            # log.info(f"Looking for next action on board\n{canonicalBoard}")

            pi = self.mcts.getActionProb(game, board, temp=temp)
            canonicalBoard = game.getCanonicalForm(board)
            sym = game.getSymmetries(canonicalBoard, pi)
            for b, p in sym:
                trainExamples.append([b.get_manual_state(), p, None]) # store the embedding of the board

            action = np.random.choice(len(pi), p=pi)
            # log.info(f"Taking action {action}")
            board = game.getNextState(board, action)

            r = game.getGameEnded(board, episodeStep-1)

            # DEBUG
            # print(print)
            # print(board)
            # print("r: ", r)

            if r != 0:
                if self.log_to_file:
                    f = open(self.filename,'a+')
                    f.write(f"Final board {board}\n")
                    f.write(f"Actions: {board.priorActions}\n")
                    f.write(f"Game over: Return {r}\n\n")
                    f.close()
                # log.info(f"Final board\n{board} with reward {r}")
                return [(x[0], x[1], r) for x in trainExamples] # update the reward for the previous moves

    def learn(self):
        """
        Performs numIters iterations with numEps episodes of self-play in each
        iteration. After every iteration, it retrains neural network with
        examples in trainExamples (which has a maximum length of maxlenofQueue).
        It then pits the new neural network against the old one and accepts it
        only if it wins >= updateThreshold fraction of games.
        """

        for i in range(1, self.args.numIters + 1):
            # bookkeeping
            log.info(f'Starting Iter #{i} ...')
            if self.log_to_file:
                f = open(self.filename,'a+')
                f.write(f'Starting Iter #{i} ...\n')
                f.close()
            # examples of the iteration
            if not self.skipFirstSelfPlay or i > 1:
                iterationTrainExamples = deque([], maxlen=self.args.maxlenOfQueue)

                for _ in tqdm(range(self.args.numEps), desc="Self Play"):
                    if self.log_to_file:
                        f = open(self.filename,'a+')
                        f.write(f'Episode #{_} ...\n')
                        f.close()
                    self.mcts = MCTS(self.nnet, self.args, self.filename)  # reset search tree
                    iterationTrainExamples += self.executeEpisode()

                # save the iteration examples to the history
                self.trainExamplesHistory.append(iterationTrainExamples)

            if len(self.trainExamplesHistory) > self.args.numItersForTrainExamplesHistory:
                log.warning(
                    f"Removing the oldest entry in trainExamples. len(trainExamplesHistory) = {len(self.trainExamplesHistory)}")
                self.trainExamplesHistory.pop(0)
            # backup history to a file
            # NB! the examples were collected using the model from the previous iteration, so (i-1)
            self.saveTrainExamples(i - 1)

            trainExamples = self.prepareTrainExamples()

            # training new network, keeping a copy of the old one
            self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            self.pnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            pmcts = MCTS(self.pnet, self.args, self.filename)

            self.nnet.train(trainExamples)
            nmcts = MCTS(self.nnet, self.args, self.filename)

            log.info('PITTING AGAINST PREVIOUS VERSION')

            arena = PlanningArena(lambda game, board: np.argmax(pmcts.getActionProb(game, board, verbose=False, temp=0)),
                                    lambda game, board: np.argmax(nmcts.getActionProb(game, board, verbose=False, temp=0)),
                                    self.game_validation, display=print, filename=self.filename, log_to_file=self.log_to_file, iter=i)
            prewards, nrewards = arena.playGames(self.args.arenaCompare, verbose=False)

            log.info('NEW/PREV WINING COUNTS : %d / %d' % (nrewards, prewards))
            if nrewards <= prewards or float(nrewards) / (prewards + nrewards) < self.args.updateThreshold:
                log.info('REJECTING NEW MODEL')
                self.nnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            else:
                log.info('ACCEPTING NEW MODEL')
                self.nnet.save_checkpoint(folder=self.args.checkpoint, filename=self.getCheckpointFile(i))
                self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='best.pth.tar')

    def prepareTrainExamples(self):

        iterationExamples = self.trainExamplesHistory[-1]
        rew = [e[2] for e in iterationExamples]
        # mean, min, std and max of the rewards
        log.info(f"REWARDS - Mean: {np.mean(rew)}, Std: {np.std(rew)}, Min: {np.min(rew)}, Max: {np.max(rew)}")

        trainExamples = []
        for e in self.trainExamplesHistory:
            trainExamples.extend(e)

        # all training examples (not only the last iteration)
        trainExamples = [(e[0], e[1], e[2]) for e in trainExamples]

        # shuffle examples before training
        shuffle(trainExamples)

        return trainExamples

    def getCheckpointFile(self, iteration):
        return 'checkpoint_' + str(iteration) + '.pth.tar'

    def saveTrainExamples(self, iteration):
        folder = self.args.checkpoint
        if not os.path.exists(folder):
            os.makedirs(folder)
        filename = os.path.join(folder, self.getCheckpointFile(iteration) + ".examples")
        with open(filename, "wb+") as f:
            Pickler(f).dump(self.trainExamplesHistory)
        f.closed

    def loadTrainExamples(self):
        modelFile = os.path.join(self.args.load_folder_file[0], self.args.load_folder_file[1])
        examplesFile = modelFile + ".examples"
        if not os.path.isfile(examplesFile):
            log.warning(f'File "{examplesFile}" with trainExamples not found!')
            r = input("Continue? [y|n]")
            if r != "y":
                sys.exit()
        else:
            log.info("File with trainExamples found. Loading it...")
            with open(examplesFile, "rb") as f:
                self.trainExamplesHistory = Unpickler(f).load()
            log.info('Loading done!')

            # examples based on the model were already collected (loaded)
            self.skipFirstSelfPlay = True
