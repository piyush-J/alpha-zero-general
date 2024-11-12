class Board():

    def __init__(self, n=3):
        "Set up initial board configuration."

        self.n = n
        # Create the empty board array with -1 filled
        self.pieces = [[-1]*self.n]*self.n # np.array([[-1]*4]*3) creates a 3x4 array with -1 filled

    # add [][] indexer syntax to the Board
    def __getitem__(self, index): 
        return self.pieces[index]

    def get_legal_moves(self):
        moves = set()  # stores the legal moves.

        # TODO: fetch from CNF file
        # for now, get all n-length moves with only one cell assigned as 1
        for y in range(self.n):
            col = [0]*self.n
            col[y] = 1
            moves.add(tuple(col))

        # convert set of tuples to list of lists
        moves = [list(move) for move in moves]

        return moves # moves are in 2-D format (x,y)

    def has_legal_moves(self):
        # if self.pieces has -1 in last cell, then there is a legal move
        if -1 in self.pieces[-1]:
            return True
        else:
            return False
    
    def is_win(self):
        # TODO: arbitrary win condition for now, remove it later and call external tool within getGameEnded()
        # check whether there is a line of ones
        win = self.n
        # check y-strips
        # for y in range(self.n):
        #     count = 0
        #     for x in range(self.n):
        #         if self[x][y]==1:
        #             count += 1 
        #     if count==win:
        #         return True
        # check x-strips
        for x in range(len(self.pieces)):
            count = 0
            for y in range(self.n):
                if self[x][y]==1:
                    count += 1
            if count==win:
                return True
        
        return False

    def execute_move(self, move):
        # add a column to the board
        column_to_add = move
        # replace the last non -1 filled column with the new column
        for i in range(len(self.pieces)):
            if -1 in self.pieces[i]:
                self.pieces[i] = column_to_add
                break
