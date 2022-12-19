# from SMTLogic import Board
from smt.SMTGame import SMTGame

def main():
    moves_str=("simplify", "smt") # , "bit-blast", "propagate-values", "ctx-simplify", "elim-uncnstr", "solve-eqs", "lia2card",  "max-bv-sharing", "nla2bv", "qfnra-nlsat", "cofactor-term-ite")
    g = SMTGame(benchmarkPath = "smt/example/debug", ext = "smt2", moves_str = moves_str)
    b0 = g.getInitBoard()
    print(b0.moves_str)
    print("File: " + b0.fPath)
    print("Current game status: 1 -- solved; 0 -- unsolved; (-1) -- give_up")
    print(g.getGameEnded(b0))
    print("Current goal embedding")
    print(g.getEmbedding(b0))
    print("Current goal manual state representation: [#assertions, #constants]")
    print(g.getManualEmbedding(b0))
    print("Next legal moves:")
    moveLst = g.getValidMoves(b0)
    print(moveLst)
    # nextMove = moveLst[0] # Talk with Piyush: this way of referring to a tactic is incorrect
    nextMove = 0
    print("Apply tactic: " + b0.moves_str[nextMove])
    print(b0.moves_str)
    b1 = g.getNextState(b0, nextMove)

    print("rlimit for this tactic application")
    print(b1.accRLimit)

    print("Current game status: 1 -- solved; 0 -- unsolved; (-1) -- give_up")
    print(g.getGameEnded(b1))
    nextMove = 1
    print("Apply tactic: " + b1.moves_str[nextMove])
    b2 = g.getNextState(b1, nextMove)
    print("rlimit for this tactic application")
    print(b2.accRLimit)
    print("Current game status: 1 -- solved; 0 -- unsolved; (-1) -- give_up")
    print(g.getGameEnded(b2))
    bd = g.getInitBoard() # a new instance for give up
    print("File: " + bd.fPath)
    for i in range (3):
        bd = g.getNextState(bd, 0) # 1 would result in [False] - is win
        print(g.getEmbedding(bd))
        print(bd.priorActions)
        print(bd)
    print("After applying 3 (limit) times of 'simplify' to a new instance of the formula")
    print("Current game status: 1 -- solved; 0 -- unsolved; (-1) -- give_up")
    print(g.getGameEnded(bd))

if __name__ == "__main__":
    main()
