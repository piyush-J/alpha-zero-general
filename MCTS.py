import logging
import math
import time

import numpy as np

EPS = 1e-8

log = logging.getLogger(__name__)


class MCTS():
    """
    This class handles the MCTS tree.
    """

    def __init__(self, nnet, args, filename):
        # self.game = game.get_copy()
        self.filename = filename
        self.nnet = nnet
        self.args = args
        # self.context = cntx
        self.log_to_file = self.args.log_to_file
        self.Qsa = {}  # stores Q values for s,a (as defined in the paper)
        self.Nsa = {}  # stores #times edge s,a was visited
        self.Ns = {}  # stores #times board s was visited
        self.Ps = {}  # stores initial policy (returned by neural net)
        self.Es = {}  # stores game.getGameEnded ended for board s
        self.Vs = {}  # stores game.getValidMoves for board s

    def getActionProb(self, game, board, temp=1, verbose=False):
        """
        This function performs numMCTSSims simulations of MCTS starting from
        canonicalBoard.

        Returns:
            probs: a policy vector where the probability of the ith action is
                   proportional to Nsa[(s,a)]**(1./temp)
        """
        canonicalBoard = game.getCanonicalForm(board)
        with open(self.filename,'a+') as f:
            f.write(f"\ngetActionprob() starts:\n")
        for i in range(self.args.numMCTSSims):
            if self.log_to_file:
                with open(self.filename,'a+') as f:
                    f.write(f"Start Sim no. {i}\n")
            self.search(game, canonicalBoard)

        s = game.stringRepresentation(canonicalBoard)
        counts = [self.Nsa[(s, a)] if (s, a) in self.Nsa else 0 for a in range(game.getActionSize())]
        if self.log_to_file:
            f = open(self.filename,'a+')
            f.write(f"After search, counts: {counts}\n")
            f.close()

        if temp == 0:
            bestAs = np.array(np.argwhere(counts == np.max(counts))).flatten()
            bestA = np.random.choice(bestAs)
            probs = [0] * len(counts)
            probs[bestA] = 1
            return probs

        counts = [x ** (1. / temp) for x in counts]
        if self.log_to_file:
            f = open(self.filename,'a+')
            f.write(f"After temp, counts: {counts}\n")
            f.close()
        counts_sum = float(sum(counts))
        probs = [x / counts_sum for x in counts]
        return probs

    def search(self, game, canonicalBoard, verbose=False, level=0):
        """
        This function performs one iteration of MCTS. It is recursively called
        till a leaf node is found. The action chosen at each node is one that
        has the maximum upper confidence bound as in the paper.

        Once a leaf node is found, the neural network is called to return an
        initial policy P and a value v for the state. This value is propagated
        up the search path. In case the leaf node is a terminal state, the
        outcome is propagated up the search path. The values of Ns, Nsa, Qsa are
        updated.

        NOTE: the return values are the negative of the value of the current
        state. This is done since v is in [-1,1] and if v is the value of a
        state for the current player, then its value is -v for the other player.

        Returns:
            v: the negative of the value of the current canonicalBoard
        """

        start_time = time.time()
        #print("1: ", time.time()-start_time)

        s = game.stringRepresentation(canonicalBoard)
        if self.log_to_file:
            with open(self.filename,'a+') as f:
                f.write(f"\nStart search with {s}\n")

        if verbose:
            log.info(f"At level {level}\n{s}")

        if s not in self.Es: # STEP 2: EXPANSION
            if verbose:
                log.info(f"Node not yet seen\n{s}")
            if self.log_to_file:
                with open(self.filename,'a+') as f:
                    f.write(f"\nNode not yet seen\n")
            self.Es[s] = game.getGameEnded(canonicalBoard)

        #print("2: ", time.time()-start_time)

        if self.Es[s] != 0: # STEP 4: BACKPROPAGATION
            # terminal node
            if self.log_to_file:
                f = open(self.filename,'a+')
                f.write(f"Search reach final board {canonicalBoard}\n")
                f.write(f"Actions: {canonicalBoard.priorActions}\n")
                f.write(f"Game over: Return {game.getGameEnded(canonicalBoard, level)}\n\n")
                f.close()
            # if verbose:
            #     log.info(f"Node is terminal node, reward is {self.Es[s]}\n{s}")
            return game.getGameEnded(canonicalBoard, level)
            # return self.Es[s] - 0.01*level - 0.01*canonicalBoard.get_time() # penalizing for number of steps and time here (because the same end state can appear at different times)

        #print("3: ", time.time()-start_time)

        if s not in self.Ps: # STEP 3: ROLLOUT or SIMULATION (use NN to predcit the value, i.e., the end reward to be backpropagated)
            # leaf node
            if verbose:
                log.info(f"Node is leaf node, using NN to predict value for\n{s}")
            with open(self.filename,'a+') as f:
                f.write(f"Node is leaf node, using NN to predict value\n")
            self.Ps[s], v = self.nnet.predict(canonicalBoard.get_manual_state()) # plays a role in calculating UCB too
            valids = game.getValidMoves(canonicalBoard)
            self.Ps[s] = self.Ps[s] * valids  # masking invalid moves
            sum_Ps_s = np.sum(self.Ps[s])
            if sum_Ps_s > 0:
                self.Ps[s] /= sum_Ps_s  # renormalize
            else:
                # if all valid moves were masked make all valid moves equally probable

                # NB! All valid moves may be masked if either your NNet architecture is insufficient or you've get overfitting or something else.
                # If you have got dozens or hundreds of these messages you should pay attention to your NNet and/or training process.
                log.error("All valid moves were masked, doing a workaround.")
                self.Ps[s] = self.Ps[s] + valids
                self.Ps[s] /= np.sum(self.Ps[s])

            self.Vs[s] = valids
            self.Ns[s] = 0
            return v # STEP 4: BACKPROPAGATION

        valids = self.Vs[s]
        cur_best = -float('inf')
        best_act = -1

        # pick the action with the highest upper confidence bound
        if verbose:
            log.info(f"Pick an action")
        for a in range(game.getActionSize()-1): # STEP 1: SELECTION
            if valids[a]:
                if (s, a) in self.Qsa:
                    if verbose:
                        log.info(f"Exists in Qsa")
                    u = self.Qsa[(s, a)] + self.args.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s]) / (
                            1 + self.Nsa[(s, a)])
                else:
                    if verbose:
                        log.info(f"Does not exist in Qsa")
                    u = self.args.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s] + EPS)  # Q = 0 ?

                if u > cur_best:
                    cur_best = u
                    best_act = a

        a = best_act
        #TODO: see why this is needed - is it because of the recursion? creating a copy because the game is modified in-place?
        game_copy = game.get_copy()
        next_s = game_copy.getNextState(canonicalBoard, a)
        # next_s = self.game.getCanonicalForm(next_s)
        if verbose:
            log.info(f"Non-leaf node, considering action {a} resulting in {next_s}\n")

        if self.log_to_file:
            with open(self.filename,'a+') as f:
                f.write(f"Non-leaf node, considering action {a} resulting in {next_s}\n")
        #print("6: ", time.time()-start_time)

        v = self.search(game_copy, next_s, level=level+1)

        if (s, a) in self.Qsa:
            self.Qsa[(s, a)] = (self.Nsa[(s, a)] * self.Qsa[(s, a)] + v) / (self.Nsa[(s, a)] + 1)
            self.Nsa[(s, a)] += 1

        else:
            self.Qsa[(s, a)] = v
            self.Nsa[(s, a)] = 1

        self.Ns[s] += 1
        return v
