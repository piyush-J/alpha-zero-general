#!/bin/bash
#SBATCH --cpus-per-task=4
#SBATCH --mem=32000M       
#SBATCH --account=def-vganesh
#SBATCH --mail-type=ALL
#SBATCH --mail-user=piyush.jha@uwaterloo.ca
#SBATCH --time=0-01:00      # time (DD-HH:MM)
#SBATCH --output=exptResults/run-%N-%j.out  # %N for node name, %j for jobID

module load cuda gcc python/3.10
source ~/alphacube_env/bin/activate

wandb login 286a4c4f6a0afc9dd24b3846168a3ff1fd9f1a3e
wandb online

cat /proc/cpuinfo | grep 'model name' | uniq

python -u main.py
