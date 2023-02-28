import multiprocessing
import numpy as np
import time

class Runner(multiprocessing.Process):

    def __init__(self, nnet, board, total_timeout, tactic_timeout, queue): # may do something to log to a file later
        multiprocessing.Process.__init__(self)

        # self._stop_event = threading.Event()
        self.q = queue
        self.initialBoard = board # may not need this
        self.curBoard = self.initialBoard
        self.nnet = nnet
        self.total_timeout = total_timeout
        self.tactic_timeout = tactic_timeout
        self.action_size = len(board.moves_str)
        self.timeed_out = False
        self.nn_time = 0
        self.solver_time = 0

    # def stop(self):
    #     self._stop_event.set()
    #
    # def stopped(self):
    #     return self._stop_event.is_set()

    # no argument for game now
    def run(self):
        time_before = time.time() # think about how to accout for time better
        priorMove = -1
        while not self.curBoard.is_win():
            acc_time = time.time() - time_before
            if acc_time > self.total_timeout:
                self.timeed_out = True
                break
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
        log_text = f"{self.curBoard}\nActions: {self.curBoard.priorActions}\n"
        if self.timeed_out:
            log_text += "timed out\n\n"
            res_tuple = (None, None, None, None, None, log_text)
        elif self.curBoard.is_win():
            if str(self.curBoard.curGoal) == "[]": res = "SAT"
            else: res = "UNSAT"
            rlimit = self.curBoard.accRLimit
            total_time = time.time() - time_before
            log_text += f"Result: {res}, rlimit: {rlimit}, time: {total_time}\n\n"
            res_tuple = (res, rlimit, total_time, self.nn_time, self.solver_time, log_text)
        else:
            log_text += "WHAT???\n\n"
            res_tuple = (None, None, None, None, None, log_text)
        self.q.put(res_tuple)

    # def collect(self):
    #     if self.is_alive():
    #         self.stop()
    #         self.join()
    #         self.timeed_out = True
    #     log_text = f"{self.curBoard}\nActions: {self.curBoard.priorActions}\n"
    #     if self.solved:
    #         log_text += f"Result: {self.result}, rlimit: {self.rlimit}, time: {self.s_time}\n\n"
    #     elif self.timeed_out:
    #         log_text += "timed out\n\n"
    #     else:
    #         log_text += "step limit? unknow?\n\n"
    #     if self.solved:
    #         return self.result, self.rlimit, self.s_time, self.nn_time, self.solver_time, log_text
    #     # explore when will come to here
    #     return None, None, None, None, None, log_text
