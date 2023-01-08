import argparse
import json
from smt.SMTGame import SMTGame
from smt.SMTGame import WIN_REWARD, NOCHANGE_REWARD, FAIL_REWARD, GIVEUP_REWARD

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('json_config', type=str, help='Json with experiment design')
    args_cl = parser.parse_args()
    config = json.load(open(args_cl.json_config, 'r'))
    print(config['tactics_config']['all_tactics'])
    print("\n basic testing")
    printingStatusStr = "Current game status: " + str(WIN_REWARD) + " -- solved; 0 -- unsolved; (" + str(GIVEUP_REWARD) + ") -- give_up; (" + str(NOCHANGE_REWARD) + ") -- nochange; (" + str(FAIL_REWARD) + ") -- fail"
    moves_str = config['tactics_config']['all_tactics']# , "bit-blast", "propagate-values", "ctx-simplify", "elim-uncnstr", "solve-eqs", "lia2card",  "max-bv-sharing", "nla2bv", "qfnra-nlsat", "cofactor-term-ite")
    g = SMTGame(benchmarkPath = "smt/example/debug", ext = "smt2", moves_str = moves_str)
    b0 = g.getInitBoard()
    print("File: " + b0.fPath)
    print(printingStatusStr)
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
    b1 = g.getNextState(b0, nextMove)

    print("rlimit for this tactic application")
    print(b1.accRLimit)

    print(printingStatusStr)
    print(g.getGameEnded(b1))
    nextMove = 1
    print("Apply tactic: " + b1.moves_str[nextMove])
    b2 = g.getNextState(b1, nextMove)
    print("rlimit for this tactic application")
    print(b2.accRLimit)
    print(printingStatusStr)
    print(g.getGameEnded(b2))

    print("\n test no change")
    bd = g.getInitBoard() # a new instance for give up
    print("File: " + bd.fPath)
    for i in range (1):
        bd = g.getNextState(bd, 0) # 1 would result in [False] - is win
        print(g.getEmbedding(bd))
        print(bd.priorActions)
        print(bd)
    print("After applying 2 (limit) times of 'simplify' to a new instance of the formula")
    print(printingStatusStr)
    print(g.getGameEnded(bd))

    print("\n test caching")
    print("Apply simply to b0 again")
    bc1 = g.getNextState(b0, 0)
    print("origin formula: ")
    print(g.getEmbedding(b0))
    print("results from caching: ")
    print(g.getEmbedding(bc1))
    print("Subsequently apply smt again: ")
    bc2 = g.getNextState(bc1, 1)
    print("results from caching: ")
    print(g.getEmbedding(bc2))
    print("initialize a new board; supposed to be the same with b0 as size is 2")
    b0again = g.getInitBoard()
    print("Apply simply:")
    b1again = g.getNextState(b0again, 0)
    print("results from caching: ")
    print(g.getEmbedding(b1again))
    b1smt = g.getNextState(b0again, 1)
    print("Apply smt to b0again:")
    print("results: ")
    print(g.getEmbedding(b1smt))



if __name__ == "__main__":
    main()
