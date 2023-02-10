import copy
import logging
from Runner import Runner

from tqdm import tqdm

log = logging.getLogger(__name__)

import functools
print = functools.partial(print, flush=True)

class PlanningArena():
    # may move val total timeout to game
    def __init__(self, nnet, game, timeout, display=None, log_file='out.txt', log_to_file=False, iter=1):
        """
        Input:
            agent 1,2: two functions that takes board as input, return action
            game: Game object
            display: a function that takes board as input and prints it (e.g.
                     display in othello/OthelloGame). Is necessary for verbose
                     mode.

        see othello/OthelloPlayers.py for an example. See pit.py for pitting
        human players/other baselines with each other.
        """
        self.nnet = nnet
        self.game = game
        self.display = display
        self.log_file = log_file
        self.log_to_file = log_to_file
        self.total_timeout = timeout
        self.iter = iter

    def playGame(self, policy, game, verbose=False):
        """
        Executes one episode of a game.
        """
        board = game.getInitBoard()
        solve_thread = Runner(policy, board, self.total_timeout, game.tactic_timeout, self.log_to_file, self.log_file)
        solve_thread.start()
        # fix which timeout is used?
        solve_thread.join(self.total_timeout)
        result, rlimit, time_res, nn_time, solver_time = solve_thread.collect()
        return result, rlimit, time_res

    def playGames(self, num, verbose=False):
        """
        Agent plays num games for self.iter of times

        Returns:
            average solved, average total time (averaged by self.iter)
        """

        f = open(self.log_file,'a+')
        f.write("Start playGames: \n")
        f.close()
        solvedLst = []
        totalTimeLst = []
        for it in range(self.iter):
            self.game.setNextFmID()
            f = open(self.log_file,'a+')
            f.write("Iteration " + str(it) + "\n")
            f.close()
            solved = 0
            totalTime = 0
            for i in tqdm(range(num), desc="Arena.playGames"):
                f = open(self.log_file,'a+')
                f.write("Instance " + str(i) + "\n")
                f.close()
                result, rlimit, time_res = self.playGame(self.nnet, self.game, verbose=verbose)
                if not result is None:
                    solved += 1
                    totalTime += time_res
            print(f"In iter {it}: {solved} instances solved with total time {totalTime}\n")
            solvedLst.append(solved)
            totalTimeLst.append(totalTime)
        avgSolved = sum(solvedLst)/len(solvedLst)
        avgTotalTime = sum(totalTimeLst)/len(totalTimeLst)
        print(f"Average: {avgSolved: .2f} instances solved with total time {avgTotalTime: .2f}\n")
        return avgSolved, avgTotalTime
