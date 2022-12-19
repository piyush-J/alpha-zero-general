#!/bin/bash

#SBATCH --cpus-per-task=6   # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --mem=32000M        # memory per node
#SBATCH --account=def-vganesh
#SBATCH --mail-type=ALL
#SBATCH --mail-user=piyush.jha@uwaterloo.ca
#SBATCH --time=0-12:00      # time (DD-HH:MM)
#SBATCH --output=run-%N-%j.out  # %N for node name, %j for jobID

module load cuda gcc python/3.7

source ~/mcts/bin/activate

export PYTHONUNBUFFERED=TRUE

python main.py
