#!/bin/bash

#SBATCH --gres=gpu:v100l:1
#SBATCH --cpus-per-task=6   # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --mem=32000M        # memory per node
#SBATCH --account=def-vganesh
#SBATCH --mail-type=ALL
#SBATCH --mail-user=piyush.jha@uwaterloo.ca
#SBATCH --time=0-18:00      # time (DD-HH:MM)
#SBATCH --output=exptResults/run-%N-%j.out  # %N for node name, %j for jobID

# Define a timestamp function
timestamp() {
  date +"%Y-%m-%d_%H-%M-%S" # current time
}

OUT_FOLD="exptResults/$(timestamp)"
mkdir -p "$OUT_FOLD"
echo "Writing output to folder $OUT_FOLD"

module load cuda gcc python/3.7

source ~/mcts/bin/activate

export PYTHONUNBUFFERED=TRUE

python ./main.py \
  > "$OUT_FOLD/out.txt" 2>&1
