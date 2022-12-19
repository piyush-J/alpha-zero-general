import copy
import logging

from tqdm import tqdm

log = logging.getLogger(__name__)

import functools
print = functools.partial(print, flush=True)


# class Arena():
#     """
#     An Arena class where any 2 agents can be pit against each other.
#     """

#     def __init__(self, player1, player2, game, display=None):
#         """
#         Input:
#             player 1,2: two functions that takes board as input, return action
#             game: Game object
#             display: a function that takes board as input and prints it (e.g.
#                      display in othello/OthelloGame). Is necessary for verbose
#                      mode.

#         see othello/OthelloPlayers.py for an example. See pit.py for pitting
#         human players/other baselines with each other.
#         """
#         self.player1 = player1
#         self.player2 = player2
#         self.game = game
#         self.display = display

#     def playGame(self, verbose=False):
#         """
#         Executes one episode of a game.

#         Returns:
#             either
#                 winner: player who won the game (1 if player1, -1 if player2)
#             or
#                 draw result returned from the game that is neither 1, -1, nor 0.
#         """
#         players = [self.player2, None, self.player1]
#         curPlayer = 1
#         board = self.game.getInitBoard()
#         it = 0
#         while self.game.getGameEnded(board, curPlayer) == 0:
#             it += 1
#             if verbose:
#                 assert self.display
#                 print("Turn ", str(it), "Player ", str(curPlayer))
#                 self.display(board)
#             action = players[curPlayer + 1](self.game.getCanonicalForm(board, curPlayer))

#             valids = self.game.getValidMoves(self.game.getCanonicalForm(board, curPlayer), 1)

#             if valids[action] == 0: # not possible because MCTS always returns a valid move
#                 log.error(f'Action {action} is not valid!')
#                 log.debug(f'valids = {valids}')
#                 assert valids[action] > 0
#             board, curPlayer = self.game.getNextState(board, curPlayer, action)
#         if verbose:
#             assert self.display
#             print("Game over: Turn ", str(it), "Result ", str(self.game.getGameEnded(board, 1)))
#             self.display(board)
#         return curPlayer * self.game.getGameEnded(board, curPlayer)

#     def playGames(self, num, verbose=False):
#         """
#         Plays num games in which player1 starts num/2 games and player2 starts
#         num/2 games.

#         Returns:
#             oneWon: games won by player1
#             twoWon: games won by player2
#             draws:  games won by nobody
#         """

#         num = int(num / 2)
#         oneWon = 0
#         twoWon = 0
#         draws = 0
#         for _ in tqdm(range(num), desc="Arena.playGames (1)"):
#             gameResult = self.playGame(verbose=verbose)
#             if gameResult == 1:
#                 oneWon += 1
#             elif gameResult == -1:
#                 twoWon += 1
#             else:
#                 draws += 1

#         self.player1, self.player2 = self.player2, self.player1

#         for _ in tqdm(range(num), desc="Arena.playGames (2)"):
#             gameResult = self.playGame(verbose=verbose)
#             if gameResult == -1:
#                 oneWon += 1
#             elif gameResult == 1:
#                 twoWon += 1
#             else:
#                 draws += 1

#         return oneWon, twoWon, draws

class PlanningArena():

    def __init__(self, agent1, agent2, game, percentile, display=None, filename='out.txt', log_to_file=False, iter=0):
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
        self.agent2 = agent2
        self.game = game
        self.display = display
        self.percentile = percentile
        self.filename = filename
        self.f = open(self.filename,'a+')
        self.log_to_file = log_to_file
        self.iter = iter

    def playGame(self, agent, game, verbose=False):
        """
        Executes one episode of a game.

        Returns:
            either
                winner: player who won the game (1 if player1, -1 if player2)
            or
                draw result returned from the game that is neither 1, -1, nor 0.
        """
        board = game.getInitBoard()
        it = 0
        while game.getGameEnded(board) == 0: # returns the reward if the game is over, else None
            it += 1
            # action = agent(game.getCanonicalForm(board))
            action = agent(game, board)
            valids = game.getValidMoves(game.getCanonicalForm(board))

            # if verbose:
            #     assert self.display
            #     print("Turn ", str(it), "Player ", str(agent))
            #     self.display(board)
            #     print("Move: ", board.moves_str[action])

            if valids[action] == 0:
                log.error(f'Action {action} is not valid!')
                log.debug(f'valids = {valids}')
                assert valids[action] > 0
            board = game.getNextState(board, action)

        if verbose:
            assert self.display
            self.display(board)
            print(f"Game over: Player {str(agent)} Turn {str(it)} Result {str(game.getGameEnded(board))}")
            print(f"Prior actions: {board.priorActions}")

        if self.log_to_file:
            self.f.write(f"Iteration {str(self.iter)}\n")
            self.f.write(str(board)+"\n")
            self.f.write(f"Game over: Player {str(agent)} Turn {str(it)} Result {str(game.getGameEnded(board))}\n")
            self.f.write(f"Prior actions: {board.priorActions}\n\n")
            
        return 1 if (game.getGameEnded(board) >= self.percentile) else 0

    def playGames(self, num, verbose=False):
        """
        Each agent plays num games
        
        Returns:
            number of rewards >= percentile for agent1
            number of rewards >= percentile for agent2
        """
        agent1Results = []
        agent2Results = []
        agent1game = copy.deepcopy(self.game) # so that every agent goes thru the same set of benchmarks # TODO: might not be needed for every game setting
        agent2game = copy.deepcopy(self.game)
        for _ in tqdm(range(num), desc="Arena.playGames"):
            agent1Results.append(self.playGame(self.agent1, agent1game, verbose=verbose))
            agent2Results.append(self.playGame(self.agent2, agent2game, verbose=verbose))
        self.f.close()
        return sum(agent1Results), sum(agent2Results)