class Runner():

    def __init__(self, nnet, filename='out.txt'): # may do something to log to a file later
        self.filename = filename
        self.nnet = nnet

    # no argument for game now
    def getActionProb(self, board):
        prob, v = self.nnet.predict(board.get_manual_state())
        return prob
