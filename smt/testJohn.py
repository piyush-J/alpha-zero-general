# from SMTLogic import Board
from SMTGame import SMTGame

def main():
    g = SMTGame() # intialize a SMT game; currently only about one formula
    b = g.getInitBoard()
    print("Current game status: 1 -- solved; 0 -- unsolved; (-1) -- give_up")
    print(g.getGameEnded(b))
    print("Current goal state representation: [#assertions, #constants]")
    print(g.getEmbedding(b))
    print("Next legal moves:")
    moveLst = g.getValidMoves(b)
    print(moveLst)
    nextMove = moveLst[0]
    print("Apply tactic: " + nextMove)
    g.getNextState(b, nextMove)
    print("Current game status: 1 -- solved; 0 -- unsolved; (-1) -- give_up")
    print(g.getGameEnded(b))
    nextMove = moveLst[1]
    print("Apply tactic: " + nextMove)
    b = g.getNextState(b, nextMove)
    print("Current game status: 1 -- solved; 0 -- unsolved; (-1) -- give_up")
    print(g.getGameEnded(b))
    bd = g.getInitBoard() # a new instance for give up
    for i in range (3):
        bd = g.getNextState(bd, "simplify")
    print("After applying 3 (limit) times of 'simplify' to a new instance of the formula")
    print("Current game status: 1 -- solved; 0 -- unsolved; (-1) -- give_up")
    print(g.getGameEnded(bd))

if __name__ == "__main__":
    main()
