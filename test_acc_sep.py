# record some examples testing whether the rlimit for applying a sequence of tactics as one tactic is the same as applying the tactics saperately in turn
from smt.SMTGame import SMTGame

def main():
    moves_str=("simplify", "smt", "qfnra-nlsat", "lia2card",  "bit-blast", "propagate-values", "ctx-simplify", "elim-uncnstr", "solve-eqs", "max-bv-sharing", "nla2bv", "cofactor-term-ite")
    g = SMTGame(benchmarkPath = "smt/example/qf_nia/AProvE/all", ext = "smt2", moves_str = moves_str) # "smt/example/qf_nia/AProvE/all" "smt/example/debug"
    print("Test 1")
    b0 = g.getInitBoard()
    b1 = g.getNextState(b0, 0)
    print(b1.is_fail())
    b2 = g.getNextState(b1, 1)
    print("accumulated rlimit for simplify + smt")
    print(b2.accRLimit)
    print(b2.is_win())

    b1p = b0.execute_move_2(0)
    print("rlimit for simplify")
    print(b1p.rlimitLastTactic)
    b2p = b1p.execute_move_2(1)
    print("rlimit for smt")
    print(b2p.rlimitLastTactic)
    print(b2p.is_win())

    print("Test 2")
    b0 = g.getInitBoard()
    b1 = g.getNextState(b0, 0)
    print(b1.is_fail())
    b2 = g.getNextState(b1, 2)
    print("accumulated rlimit for simplify + qfnra-nlsat")
    print(b2.accRLimit)
    print(b2.is_win())

    b1p = b0.execute_move_2(0)
    print("rlimit for simplify")
    print(b1p.rlimitLastTactic)
    b2p = b1p.execute_move_2(2)
    print("rlimit for qfnra-nlsat")
    print(b2p.rlimitLastTactic)
    print(b2p.is_win())

    print("Test 3")
    b0 = g.getInitBoard()
    b1 = g.getNextState(b0, 0)
    print(b1.is_fail())
    b2 = g.getNextState(b1, 3)
    print(b2.is_fail())
    b3 = g.getNextState(b2, 2)
    print("accumulated rlimit for simplify + qfnra-nlsat")
    print(b3.accRLimit)
    print(b3.is_win())

    b1p = b0.execute_move_2(0)
    print("rlimit for simplify")
    print(b1p.rlimitLastTactic)
    b2p = b1p.execute_move_2(3)
    print("rlimit for qfnra-nlsat")
    print(b2p.rlimitLastTactic)
    b3p = b2p.execute_move_2(2)
    print("rlimit for qfnra-nlsat")
    print(b3p.rlimitLastTactic)
    print(b3p.is_win())

    # print(b0.moves_str)
    # print("File: " + b0.fPath)
    # print("Current game status: 1 -- solved; 0 -- unsolved; (-1) -- give_up")
    # print(g.getGameEnded(b0))
    # print("Current goal embedding")
    # print(g.getEmbedding(b0))
    # print("Current goal manual state representation: [#assertions, #constants]")
    # print(g.getManualEmbedding(b0))
    # print("Next legal moves:")
    # moveLst = g.getValidMoves(b0)
    # print(moveLst)
    # # nextMove = moveLst[0] # Talk with Piyush: this way of referring to a tactic is incorrect
    # nextMove = 0
    # print("Apply tactic: " + b0.moves_str[nextMove])
    # print(b0.moves_str)
    # b1 = g.getNextState(b0, nextMove)
    #
    # print("rlimit for this tactic application")
    # print(b1.accRLimit)
    #
    # print("Current game status: 1 -- solved; 0 -- unsolved; (-1) -- give_up")
    # print(g.getGameEnded(b1))
    # nextMove = 1
    # print("Apply tactic: " + b1.moves_str[nextMove])
    # b2 = g.getNextState(b1, nextMove)
    # print("rlimit for this tactic application")
    # print(b2.accRLimit)
    # print("Current game status: 1 -- solved; 0 -- unsolved; (-1) -- give_up")
    # print(g.getGameEnded(b2))
    # bd = g.getInitBoard() # a new instance for give up
    # print("File: " + bd.fPath)
    # for i in range (3):
    #     bd = g.getNextState(bd, 0) # 1 would result in [False] - is win
    #     print(g.getEmbedding(bd))
    #     print(bd.priorActions)
    #     print(bd)
    # print("After applying 3 (limit) times of 'simplify' to a new instance of the formula")
    # print("Current game status: 1 -- solved; 0 -- unsolved; (-1) -- give_up")
    # print(g.getGameEnded(bd))

if __name__ == "__main__":
    main()
