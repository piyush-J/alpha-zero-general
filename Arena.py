import copy
import logging
from Runner import Runner

from tqdm import tqdm

log = logging.getLogger(__name__)

import functools
print = functools.partial(print, flush=True)

TOTAL_TIMEOUT = 60 # move it to json later

class PlanningArena():

    def __init__(self, nn1, nn2, game, display=None, log_file='out.txt', log_to_file=False, iter=0):
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
        self.nn1 = nn1
        self.nn2 = nn2
        self.game = game
        self.display = display
        self.log_file = log_file
        self.log_to_file = log_to_file
        self.iter = iter

    def playGame(self, policy, game, verbose=False):
        """
        Executes one episode of a game.
        """
        board = game.getInitBoard()
        solve_thread = Runner(policy, board, TOTAL_TIMEOUT, game.tactic_timeout, self.log_to_file, self.log_file)
        solve_thread.start()
        solve_thread.join(TOTAL_TIMEOUT)
        result, rlimit, time_res, nn_time, solver_time = solve_thread.collect()
        # while game.getGameEnded(board, it-1) == 0: # returns the reward if the game is over, else None
        #
        #     action = agent(board)
        #     valids = game.getValidMoves(game.getCanonicalForm(board))
        #
        #     if valids[action] == 0:
        #         log.error(f'Action {action} is not valid!')
        #         log.debug(f'valids = {valids}')
        #         assert valids[action] > 0
        #     board = game.getNextState(board, action)

        # write log later

        # if verbose:
        #     assert self.display
        #     self.display(board)
        #     print(f"Game over: Player {str(agent)} Turn {str(it)} Result {str(game.getGameEnded(board, it-1))}")
        #     print(f"Prior actions: {board.priorActions}")
        #
        # if self.log_to_file:
        #     self.f.write(str(board)+"\n")
        #     self.f.write(f"Actions: {board.priorActions}\n")
        #     self.f.write(f"Game over: Return {str(game.getGameEnded(board, it-1))}\n\n")

        return result, rlimit, time_res

    def playGames(self, num, verbose=False):
        """
        Each agent plays num games

        Returns:
            number of rewards >= percentile for agent1 (update later)
            number of rewards >= percentile for agent2
        """
        policy1Solved = 0
        policy2Solved = 0
        policy1TotalTime = 0
        policy2TotalTime = 0
        f = open(self.log_file,'a+')
        f.write("Agent 1 (prev): \n")
        f.close()
        for i in tqdm(range(num), desc="Arena.playGames1"):
            f = open(self.log_file,'a+')
            f.write("Iteration " + str(i) + "\n")
            f.close()
            result, rlimit, time_res = self.playGame(self.nn1, self.game, verbose=verbose)
            if not result is None:
                policy1Solved += 1
                policy1TotalTime += time_res
        self.game.setNextFmID()
        f = open(self.log_file,'a+')
        f.write("Agent 2 (new): \n")
        f.close()
        for i in tqdm(range(num), desc="Arena.playGames2"):
            f = open(self.log_file,'a+')
            f.write("Iteration " + str(i) + "\n")
            f.close()
            result, rlimit, time_res = self.playGame(self.nn2, self.game, verbose=verbose)
            if not result is None:
                policy2Solved += 1
                policy2TotalTime += time_res
        self.game.setNextFmID()
        # print("Agent1 results: " + str(agent1Results))
        # print("Agent2 results: " + str(agent2Results))
        # agent1Wins = 0
        # agent2Wins = 0
        # for i in range(num):``
        #     if agent2Results[i] > agent1Results[i]: agent2Wins += 1
        #     elif agent2Results[i] < agent1Results[i]: agent1Wins += 1

        #current use time # make it an option later
        print(f"prev NN solves: {policy1Solved} with total time {policy1TotalTime};\n new NN solves: {policy2Solved} with total time {policy2TotalTime}\n")
        newWin = 0
        if (policy2Solved > policy1Solved) or ((policy2Solved == policy1Solved) and (policy1TotalTime > policy2TotalTime)): newWin = 1
        return 0, newWin
