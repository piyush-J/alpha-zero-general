#!/bin/bash
set -x

python -u main.py "constraints_20_c_100000_2_2_0_final.simp" -order 20 -n 40 -m 190 -o "e4_20_mcts_nodepthlimit.cubes" -numMCTSSims 20 -cpuct 10 -varpen 0 > cubing_outputs/e4_20_mcts_nodepthlimit.out 2>&1;
python -u main.py "constraints_20_c_100000_2_2_0_final.simp" -order 20 -n 40 -m 190 -o "e4_20_mcts_nod_sim70.cubes" -numMCTSSims 70 -cpuct 10 -varpen 0 > cubing_outputs/e4_20_mcts_nod_sim70.out 2>&1;
python -u main.py "constraints_20_c_100000_2_2_0_final.simp" -order 20 -n 40 -m 190 -o "e4_20_mcts_nod_sim300.cubes" -numMCTSSims 300 -cpuct 10 -varpen 0 > cubing_outputs/e4_20_mcts_nod_sim300.out 2>&1;
python -u main.py "constraints_20_c_100000_2_2_0_final.simp" -order 20 -n 40 -m 190 -o "e4_20_mcts_nod_s300_c3.cubes" -numMCTSSims 300 -cpuct 3 -varpen 0 > cubing_outputs/e4_20_mcts_nod_s300_c3.out 2>&1;
python -u main.py "constraints_20_c_100000_2_2_0_final.simp" -order 20 -n 40 -m 190 -o "e4_20_mcts_nod_s300_c05.cubes" -numMCTSSims 300 -cpuct 0.5 -varpen 0 > cubing_outputs/e4_20_mcts_nod_s300_c05.out 2>&1;
python -u main.py "constraints_20_c_100000_2_2_0_final.simp" -order 20 -n 40 -m 190 -o "e4_20_mcts_nod_s300_c3_pen02.cubes" -numMCTSSims 300 -cpuct 3 -varpen 0.2 > cubing_outputs/e4_20_mcts_nod_s300_c3_pen02.out 2>&1;
python -u march_pysat_m.py "constraints_20_c_100000_2_2_0_final.simp" -n 40 -m 190 -o "e4_20_pysat_varm.cubes" > cubing_outputs/e4_20_pysat_varm.out 2>&1;

python -u main.py "constraints_21_10000_2_2_0_10_final.simp" -order 21 -n 60 -m 210 -o "e4_21_mcts_nodepthlimit.cubes" -numMCTSSims 20 -cpuct 10 -varpen 0 > cubing_outputs/e4_21_mcts_nodepthlimit.out 2>&1;
python -u main.py "constraints_21_10000_2_2_0_10_final.simp" -order 21 -n 60 -m 210 -o "e4_21_mcts_nod_sim70.cubes" -numMCTSSims 70 -cpuct 10 -varpen 0 > cubing_outputs/e4_21_mcts_nod_sim70.out 2>&1;
python -u main.py "constraints_21_10000_2_2_0_10_final.simp" -order 21 -n 60 -m 210 -o "e4_21_mcts_nod_sim300.cubes" -numMCTSSims 300 -cpuct 10 -varpen 0 > cubing_outputs/e4_21_mcts_nod_sim300.out 2>&1;
python -u main.py "constraints_21_10000_2_2_0_10_final.simp" -order 21 -n 60 -m 210 -o "e4_21_mcts_nod_s300_c3.cubes" -numMCTSSims 300 -cpuct 3 -varpen 0 > cubing_outputs/e4_21_mcts_nod_s300_c3.out 2>&1;
python -u main.py "constraints_21_10000_2_2_0_10_final.simp" -order 21 -n 60 -m 210 -o "e4_21_mcts_nod_s300_c05.cubes" -numMCTSSims 300 -cpuct 0.5 -varpen 0 > cubing_outputs/e4_21_mcts_nod_s300_c05.out 2>&1;
python -u main.py "constraints_21_10000_2_2_0_10_final.simp" -order 21 -n 60 -m 210 -o "e4_21_mcts_nod_s300_c3_pen02.cubes" -numMCTSSims 300 -cpuct 3 -varpen 0.2 > cubing_outputs/e4_21_mcts_nod_s300_c3_pen02.out 2>&1;
python -u march_pysat_m.py "constraints_21_10000_2_2_0_10_final.simp" -n 60 -m 210 -o "e4_21_pysat_varm.cubes" > cubing_outputs/e4_21_pysat_varm.out 2>&1;
