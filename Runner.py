import threading
import numpy as np
import time

class Runner(threading.Thread):

    def __init__(self, nnet, board, total_timeout, tactic_timeout, log_to_file=False, log_file='out.txt'): # may do something to log to a file later
        threading.Thread.__init__(self)
        self._stop_event = threading.Event()
        self.initialBoard = board # may not need this
        self.curBoard = self.initialBoard
        self.nnet = nnet
        self.total_timeout = total_timeout
        self.tactic_timeout = tactic_timeout
        self.action_size = len(board.moves_str)
        self.solved = False
        self.log_to_file = log_to_file
        self.log_file = log_file
        self.timeed_out = False
        self.nn_time = 0
        self.solver_time = 0

    def stop(self):
        self._stop_event.set()

    def stopped(self):
        return self._stop_event.is_set()

    # no argument for game now
    def run(self):
        time_before = time.time() # think about how to accout for time better
        priorMove = -1
        while not self.curBoard.is_done():
            if self.stopped(): break
            nn_before_time = time.time()
            prob, v = self.nnet.predict(self.curBoard.get_manual_state())
            move = np.argmax(prob)
            if move == priorMove: # may improve the desgin here
                move = np.argsort(prob)[-2]
            assert(move < self.action_size)
            priorMove = move
            # no valid check since now all actions are valid
            nn_after_time = time.time()
            self.nn_time += nn_after_time - nn_before_time
            self.curBoard = self.curBoard.execute_move(move, self.tactic_timeout)
            self.solver_time += time.time() - nn_after_time
        if self.curBoard.is_win():
            self.solved = True
            if str(self.curBoard.curGoal) == "[]": self.result  = "SAT"
            else: self.result  = "UNSAT"
            self.rlimit = self.curBoard.accRLimit
            self.s_time = time.time() - time_before

    def collect(self):
        if self.is_alive():
            self.stop()
            self.join()
            self.timeed_out = True
        if self.log_to_file:
            f = open(self.log_file,'a+')
            f.write(str(self.curBoard)+"\n")
            f.write(f"Actions: {self.curBoard.priorActions}\n")
            if self.solved:
                f.write(f"Result: {self.result}, rlimit: {self.rlimit}, time: {self.s_time}\n\n")
            elif self.timeed_out:
                f.write("timed out\n\n")
            else:
                f.write("step limit? unknow?\n\n")
            f.close()
        if self.solved:
            return self.result, self.rlimit, self.s_time, self.nn_time, self.solver_time
        # explore when will come to here
        return None, None, None, None, None
