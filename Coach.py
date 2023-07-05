import logging
import os
import pickle
import sys
from collections import deque
from pickle import Pickler, Unpickler
from random import shuffle
from matplotlib import pyplot as plt

import numpy as np
from tqdm import tqdm
import wandb

from Arena import PlanningArena
from MCTS import MCTS

import itertools

log = logging.getLogger(__name__)


class Coach():
    """
    This class executes the self-play + learning. It uses the functions defined
    in Game and NeuralNet. args are specified in main.py.
    """

    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.pnet = self.nnet.__class__(self.game)  # the competitor network
        self.args = args
        self.all_logging_data = []
        self.nn_iteration = None
        self.mcts = MCTS(self.nnet, self.args, self.all_logging_data, self.nn_iteration)
        self.trainExamplesHistory = []  # history of examples from args.numItersForTrainExamplesHistory latest iterations
        self.skipFirstSelfPlay = False  # can be overriden in loadTrainExamples()
        self.leaf_counter = 0

    def DFSUtil(self, game, board, level, trainExamples, all_cubes):
        # TODO: Incorporate canonicalBoard & symmetry appropriately when required in the future
        # canonicalBoard = game.getCanonicalForm(board)
        # sym = game.getSymmetries(canonicalBoard, pi)
        # for b, p in sym:
        #     trainExamples.append([b.get_state(), p, None])
        
        # visited.add(v) # no need if we are using a tree

        reward_now = game.getGameEnded(board)
        if reward_now: # reward is not None, i.e., game over
            flattened_list = itertools.chain.from_iterable(board.prior_actions)
            all_cubes.append(flattened_list)
            self.leaf_counter += 1
            if self.args.debugging: log.info(f"Leaf node: {self.leaf_counter} with state: {board}")
            return reward_now # only leaves have rewards & leaves don't have neighbors
        else: # None
            reward_now = 0 # initialize reward for non-leaf nodes
        # Non-leaf nodes
        temp = int(level < self.args.tempThreshold)
        if self.args.debugging: log.info(f"-----------------------------------\nDFS level: {level}")
        pi = self.mcts.getActionProb(game, board, temp=temp)
        valids = game.getValidMoves(board)

        a = np.random.choice(len(pi), p=pi)
        if self.args.debugging: log.info(f"DFS best action is {a} with pi = {pi[a]:.3f}, max pi value {max(pi):.3f}, same pi count = {sum(np.array(pi) == pi[a])}")
        game_copy_dir1 = game.get_copy()
        next_s_dir1 = game_copy_dir1.getNextState(board, a)

        comp_a = board.get_complement_action(a) # complement of the literal
        game_copy_dir2 = game.get_copy()
        next_s_dir2 = game_copy_dir2.getNextState(board, comp_a)

        assert valids[a] and valids[comp_a], "Invalid action chosen by MCTS"

        for game_n, neighbour in zip((game_copy_dir1, game_copy_dir2), (next_s_dir1, next_s_dir2)): 
            reward_now += self.DFSUtil(game_n, neighbour, level+1, trainExamples, all_cubes)
        reward_now = reward_now/2 # average reward of the two children
        
        trainExamples.append([board.get_state(), pi, reward_now]) # after all children are visited, add a reward to the current node
        return reward_now # return the reward to the parent

    def executeEpisode(self):
        """
        This function executes one episode of self-play.

        It uses a temp=1 if episodeStep < tempThreshold, and thereafter
        uses temp=0.

        Returns:
            trainExamples: a list of examples of the form (canonicalBoard, currPlayer, pi,v)
                           pi is the MCTS informed policy vector, v is +1 if
                           the player eventually won the game, else -1.
        """
        trainExamples = []
        all_cubes = []
        game = self.game.get_copy()
        board = game.getInitBoard()

        self.leaf_counter = 0
        r = self.DFSUtil(game, board, level=1, trainExamples=trainExamples, all_cubes=all_cubes)

        if self.args.MCTSmode == 0:
            arena_cubes = [list(map(str, l)) for l in all_cubes]
            if os.path.exists("arena_cubes.txt"):
                log.info("arena_cubes.txt already exists. Replacing old file!")
            f = open("arena_cubes.txt", "w")
            f.writelines(["a " + " ".join(l) + " 0\n" for l in arena_cubes])
            f.close()

            log.info("Saved cubes to file")
            print("Reward: ", r)
            with open('trainExamples.pkl', 'wb') as f:
                pickle.dump(trainExamples, f)
            print("Saved Training examples to trainExamples.pkl")
        return trainExamples

    def nolearnMCTS(self):
        self.mcts = MCTS(self.nnet, self.args, self.all_logging_data, self.nn_iteration)  # reset search tree every episode
        # if os.path.exists("mcts_cache.pkl"):
        #     with open('mcts_cache.pkl', 'rb') as f:
        #         self.mcts.cache_data = pickle.load(f)
        self.executeEpisode()
        # with open('mcts_cache.pkl', 'wb') as f:
        #     pickle.dump(self.mcts.cache_data, f)

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
            self.NN_iteration = i
            # examples of the iteration
            if not self.skipFirstSelfPlay or i > 1:
                iterationTrainExamples = deque([], maxlen=self.args.maxlenOfQueue)

                for _ in tqdm(range(self.args.numEps), desc="Self Play"):
                    # TODO: can be parallelized
                    self.all_logging_data += self.mcts.data
                    self.mcts = MCTS(self.nnet, self.args, self.all_logging_data, self.nn_iteration)  # reset search tree every episode
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

            trainExamples, perc = self.prepareTrainExamples(i)

            # training new network, keeping a copy of the old one
            self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            self.pnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            pmcts = MCTS(self.pnet, self.args, self.all_logging_data, self.nn_iteration)

            self.nnet.train(trainExamples)
            nmcts = MCTS(self.nnet, self.args, self.all_logging_data, self.nn_iteration)

            log.info('PITTING AGAINST PREVIOUS VERSION')
            random_action_agent = lambda game, board: np.random.choice([board.lits2var[l] for l in board.get_legal_literals()])
            arena = PlanningArena(lambda game, board: np.argmax(pmcts.getActionProb(game, board, verbose=False, temp=0)),
                                    lambda game, board: np.argmax(nmcts.getActionProb(game, board, verbose=False, temp=0)), self.game, perc, i)#, display=print)
            prewards, nrewards = arena.playGames(self.args.arenaCompare, verbose=False, vsRandom=None) # pass random_action_agent

            log.info('NEW/PREV REWARDS : %d / %d' % (nrewards, prewards))
            wandb.log({"new_rewards": nrewards, "prev_rewards": prewards, "iteration": i})
            if nrewards <= prewards: # or float(nrewards) / (prewards + nrewards) < self.args.updateThreshold:
                log.info('REJECTING NEW MODEL')
                self.nnet.load_checkpoint(folder=self.args.checkpoint, filename='temp.pth.tar')
            else:
                log.info('ACCEPTING NEW MODEL')
                self.nnet.save_checkpoint(folder=self.args.checkpoint, filename=self.getCheckpointFile(i))
            self.nnet.save_checkpoint(folder=self.args.checkpoint, filename='best.pth.tar')
            self.mcts.nnet = self.nnet # update the search tree model with the new model

    def prepareTrainExamples(self, iteration):

        iterationExamples = self.trainExamplesHistory[-1]
        rew = [e[2] for e in iterationExamples]
        
        # mean, min, std and max of the rewards
        log.info(f"REWARDS - Mean: {np.mean(rew)}, Std: {np.std(rew)}, Min: {np.min(rew)}, Max: {np.max(rew)}")
        plt.boxplot(rew)
        plt.savefig("boxplot_rew.png")
        wandb.log({"Rewards (train set)": wandb.Image("boxplot_rew.png")})

        perc = np.percentile(rew, 90)
        log.info(f"Percentile is {perc}")
        wandb.log({"mean_reward_tr": np.mean(rew), "std_reward_tr": np.std(rew), "min_reward_tr": np.min(rew), "max_reward_tr": np.max(rew), "percentile_tr": perc, "iteration": iteration})

        trainExamples = []
        for e in self.trainExamplesHistory:
            trainExamples.extend(e)

        # all training examples (not only the last iteration)
        trainExamples = [(e[0], e[1], e[2]) for e in trainExamples]

        # shuffle examples before training
        shuffle(trainExamples)

        return trainExamples, perc

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
