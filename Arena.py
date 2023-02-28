import copy
import logging
from Runner import Runner

from tqdm import tqdm
import multiprocessing
import time

log = logging.getLogger(__name__)

import functools
print = functools.partial(print, flush=True)

class PlanningArena():
    # may move val total timeout to game
    def __init__(self, nnet, game, log_to_file, log_file, iter, val_batch):
        """
        Input:
            agent 1,2: two functions that takes board as input, return action
            game: Game object

        see othello/OthelloPlayers.py for an example. See pit.py for pitting
        human players/other baselines with each other.
        """
        self.nnet = nnet
        self.game = game
        self.log_file = log_file
        self.log_to_file = log_to_file
        self.total_timeout = game.total_timeout
        self.iter = iter
        self.val_batch = val_batch

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
            f = open(self.log_file,'a+')
            f.write("Iteration " + str(it) + "\n")
            f.close()
            solved = 0
            totalTime = 0
            q = multiprocessing.Queue()
            for i in tqdm(range(0, num, self.val_batch), desc="Arena.playGames"):
                batch_instance_ids = range(i,min(i+self.val_batch, num))
                processes = []
                for id in batch_instance_ids:
                    board = self.game.getInitBoard(id)
                    processes.append(Runner(self.nnet, board, self.total_timeout, self.game.tactic_timeout, q))
                for process in processes:
                    process.start()
                t1 = time.time()
                for process in processes:
                    t2 = time.time()
                    process.join(max(0.0001, self.total_timeout + self.game.tactic_timeout - (t2-t1) + 10)) # the timeout used in this join is not for real total timeout; just prevent deadlock
                for process in processes:
                    if process.is_alive(): process.terminate()
            while (not q.empty()):
                result, rlimit, time_res, nn_time, solver_time, log_info = q.get()
                if self.log_to_file:
                    with open(self.log_file,'a+') as f: f.write(log_info)
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
