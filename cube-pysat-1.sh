#!/bin/bash
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G       
#SBATCH --account=def-vganesh
#SBATCH --mail-type=ALL
#SBATCH --mail-user=piyush.jha@uwaterloo.ca
#SBATCH --time=0-00:10      # time (DD-HH:MM)
#SBATCH --output=cubing_outputs/e4_18_pysat_var-%N-%j.out  # %N for node name, %j for jobID

module load python/3.10
source ~/alphacube_env/bin/activate

python -u march_pysat.py "constraints_18_c_100000_2_2_0_final.simp" -n 20 -m 153 -o "e4_18_pysat_var.cubes"
