#!/bin/bash
#SBATCH --cpus-per-task=4
#SBATCH --mem=12000M       
#SBATCH --account=def-vganesh
#SBATCH --mail-type=ALL
#SBATCH --mail-user=piyush.jha@uwaterloo.ca
#SBATCH --time=0-01:00      # time (DD-HH:MM)
#SBATCH --output=cubing_outputs/e4_18_mcts_def-%N-%j.out  # %N for node name, %j for jobID

module load cuda gcc python/3.10
source ~/alphacube_env/bin/activate

# wandb login 286a4c4f6a0afc9dd24b3846168a3ff1fd9f1a3e
# wandb online

# cat /proc/cpuinfo | grep 'model name' | uniq

python -u main.py "constraints_18_c_100000_2_2_0_final.simp" -order 18 -n 20 -m 153 -o "e4_18_mcts_def.cubes"
