import re
import subprocess

result = subprocess.run(['../PhysicsCheck/gen_cubes/march_cu/march_cu', 
                         'constraints_17_c_100000_2_2_0_final.simp',
                         '-o',
                         'mcts_logs/tmp.cubes', 
                         '-d', '1', '-m', '136'], capture_output=True, text=True)
output = result.stdout

# two groups enclosed in separate ( and ) bracket
march_var_score_dict = dict(re.findall(r"alphasat: variable (\d+) with score (\d+)", output))
march_var_score_dict = {int(k):int(v) for k,v in march_var_score_dict.items()}

march_var_node_score_list = list(map(int, re.findall(r"selected (\d+) at (\d+) with diff (\d+)", output)[0]))

print(march_var_score_dict)
print(march_var_node_score_list)