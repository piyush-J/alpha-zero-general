# from SMTLogic import Board
from smt.SMTGame import SMTGame

def main():
    g = SMTGame(formulaPath = "smt/example/qfniaex.smt2") # intialize a SMT game; currently only about one formula
    b = g.getInitBoard()
    print("Current game status: 1 -- solved; 0 -- unsolved; (-1) -- give_up")
    print(g.getGameEnded(b))
    print("Current goal state representation: [#assertions, #constants]")
    print(g.getEmbedding(b))
    print("Next legal moves:")
    moveLst = g.getValidMoves(b)
    print(moveLst)
    nextMove = moveLst[0]
    print("Apply tactic: " + b.moves_str[nextMove])
    g.getNextState(b, nextMove)
    print("Current game status: 1 -- solved; 0 -- unsolved; (-1) -- give_up")
    print(g.getGameEnded(b))
    nextMove = moveLst[1]
    print("Apply tactic: " + b.moves_str[nextMove])
    b = g.getNextState(b, nextMove)
    print("Current game status: 1 -- solved; 0 -- unsolved; (-1) -- give_up")
    print(g.getGameEnded(b))
    bd = g.getInitBoard() # a new instance for give up
    for _ in range (3):
        bd = g.getNextState(bd, b.moves_str.index("simplify"))
    print("After applying 3 (limit) times of 'simplify' to a new instance of the formula")
    print("Current game status: 1 -- solved; 0 -- unsolved; (-1) -- give_up")
    print(g.getGameEnded(bd))

if __name__ == "__main__":
    main()
