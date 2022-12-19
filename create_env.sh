module load cuda gcc python/3.7

virtualenv --no-download --clear ~/mcts

source ~/mcts/bin/activate

pip install -r requirements_cc.txt