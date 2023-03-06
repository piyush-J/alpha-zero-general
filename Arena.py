import logging

from tqdm import tqdm
import numpy as np
import wandb

log = logging.getLogger(__name__)

class PlanningArena():

    def __init__(self, agent1, agent2, game, percentile, iter, display=None):
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
        self.agent1 = agent1
        self.agent2 = agent2 # new agent
        self.game = game
        self.display = display
        self.percentile = percentile
        self.iter = iter

    def calcAndLogMetrics(self, agent1LeafTimes, agentString, newagent):
        avg_count1 = np.array([len(t) for t in agent1LeafTimes])
        avg_mean1 = np.array([np.mean(t) for t in agent1LeafTimes])
        avg_max1 = np.array([np.max(t) for t in agent1LeafTimes])
        avg_min1 = np.array([np.min(t) for t in agent1LeafTimes])
        avg_std1 = np.array([np.std(t) for t in agent1LeafTimes])

        prob1 = [a/a.sum() for a in agent1LeafTimes]
        entropy1 = np.array([(-p*np.log2(p)).mean() for p in prob1])

        if newagent:
            wandb.log({"count": np.mean(avg_count1), 
                    "entropy": np.mean(entropy1),
                    "mean": np.mean(avg_mean1),
                    "std": np.mean(avg_std1),
                    "min": np.mean(avg_min1),
                    "max": np.mean(avg_max1),
                    "iteration": self.iter})

        log.info(f"LEAF REWARDS for {agentString} - \
                 Count: {np.mean(avg_count1)}, \
                 Entropy: {np.mean(entropy1)}, \
                 Mean: {np.mean(avg_mean1)}, \
                 Std: {np.mean(avg_std1)}, \
                 Min: {np.mean(avg_min1)}, \
                 Max: {np.mean(avg_max1)}")

    def DFSUtil(self, game, board, level, agent, solver_time):
        # TODO: Incorporate canonicalBoard & symmetry appropriately when required in the future
        # canonicalBoard = game.getCanonicalForm(board)
        # sym = game.getSymmetries(canonicalBoard, pi)
        # for b, p in sym:
        #     trainExamples.append([b.get_state(), p, None])
        
        # visited.add(v) # no need if we are using a tree

        reward_now = game.getGameEnded(board)
        if reward_now: # reward is not None, i.e., game over
            if board.is_giveup():
                solver_time.append(-reward_now*10)
            return reward_now # only leaves have rewards & leaves don't have neighbors

        # Non-leaf nodes
        valids = game.getValidMoves(board)

        a = agent(game, board)
        game_copy_dir1 = game.get_copy()
        next_s_dir1 = game_copy_dir1.getNextState(board, a)

        comp_a = board.get_complement_action(a) # complement of the literal
        game_copy_dir2 = game.get_copy()
        next_s_dir2 = game_copy_dir2.getNextState(board, comp_a)

        assert valids[a] and valids[comp_a], "Invalid action chosen by MCTS"

        for game_n, neighbour in zip((game_copy_dir1, game_copy_dir2), (next_s_dir1, next_s_dir2)): 
            reward_now += self.DFSUtil(game_n, neighbour, level+1, agent, solver_time)
        
        return reward_now # return the reward to the parent

    def playGame(self, agent, verbose=False):
        """
        Executes one episode of a game.

        Returns:
            either
                winner: player who won the game (1 if player1, -1 if player2)
            or
                draw result returned from the game that is neither 1, -1, nor 0.
        """
        game = self.game.get_copy()
        board = game.getInitBoard()

        solver_time = [] # solver time in seconds at leaf nodes (when game is in giveup state)
        
        rew = self.DFSUtil(game, board, level=1, agent=agent, solver_time=solver_time)

        return rew, np.array(solver_time)

    def playGames(self, num, verbose=False, vsRandom=None):
        """
        Each agent plays num games
        
        Returns:
            agent1: number of rewards > percentile
            agent2: number of rewards > percentile
        """
        agent1Results = []
        agent1LeafTimes = []
        agent2Results = []
        agent2LeafTimes = []
        randomResults = []
        randomLeafTimes = []
        for _ in tqdm(range(num), desc="Arena.playGames"):
            rew1, leaf1 = self.playGame(self.agent1, verbose=verbose)
            agent1Results.append(rew1)
            agent1LeafTimes.append(leaf1)
            rew2, leaf2 = self.playGame(self.agent2, verbose=verbose)
            agent2Results.append(rew2)
            agent2LeafTimes.append(leaf2)
            if vsRandom:
                rew3, leaf3 = self.playGame(vsRandom, verbose=verbose)
                randomResults.append(rew3)
                randomLeafTimes.append(leaf3)

        self.calcAndLogMetrics(agent1LeafTimes, "Agent1", newagent=False)
        self.calcAndLogMetrics(agent2LeafTimes, "Agent2", newagent=True)
        if vsRandom:
            self.calcAndLogMetrics(randomLeafTimes, "Random", newagent=False)
            log.info(f"Reward for agent1 = {sum(agent1Results)}, agent2 = {sum(agent2Results)}, random = {sum(randomResults)}")
        return sum(agent1Results), sum(agent2Results)