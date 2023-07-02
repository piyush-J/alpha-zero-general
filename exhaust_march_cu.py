

import operator
import re
import subprocess
import pickle
from pysat.formula import CNF


order = 17
valid_literals = None
prob = None
march_pos_lit_score_dict = None
current_metric_val = None

org_filename = "constraints_17_c_100000_2_2_0_final.simp"
filename = "tmp.cnf"
edge_vars = order*(order-1)//2 

all_results = {}
all_results_nested = {}

cnf = CNF(from_file=org_filename)
max_metric_val = len(cnf.clauses) # maximum possible value of the metric (unweighted)

cnf.to_file(filename)

# ../PhysicsCheck/gen_cubes/march_cu/march_cu tmp.cnf -o tmp.cubes -d 1 -m 136
result = subprocess.run(['../PhysicsCheck/gen_cubes/march_cu/march_cu', 
                        filename,
                        '-o',
                        'tmp.cubes', 
                        '-d', '1', '-m', str(edge_vars)], capture_output=True, text=True)

output = result.stdout

# two groups enclosed in separate ( and ) bracket
march_pos_lit_score_dict = dict(re.findall(r"alphasat: variable (\d+) with score (\d+)", output))
march_pos_lit_score_dict = {int(k):float(v) for k,v in march_pos_lit_score_dict.items()}
# all_results.update(march_pos_lit_score_dict) # merge dict

res = None
if len(march_pos_lit_score_dict) == 0:
    unsat_check = re.findall(r"c number of cubes (\d+), including (\d+) refuted leaf", output)
    if unsat_check[0][0] == unsat_check[0][1] in output:
        assert len(unsat_check) == 1
        res = 0
        print("Unsat!")
    elif "SATISFIABLE" in output:
        res = 1
        print("Sat!")
    else:
        print("Unknown result with empty dict!")
        print(output)
        print("-"*50)

pos_keys = list(march_pos_lit_score_dict.keys())
neg_keys = [-k for k in pos_keys]
all_keys = pos_keys + neg_keys

for k in all_keys:
    cnf = CNF(from_file=org_filename)

    print(k)
    cnf.append([k])
    all_results_nested[k] = {}

    cnf.to_file(filename)

    # ../PhysicsCheck/gen_cubes/march_cu/march_cu tmp.cnf -o tmp.cubes -d 1 -m 136
    result = subprocess.run(['../PhysicsCheck/gen_cubes/march_cu/march_cu', 
                            filename,
                            '-o',
                            'tmp.cubes', 
                            '-d', '1', '-m', str(edge_vars)], capture_output=True, text=True)

    output = result.stdout

    march_pos_lit_score_dict2 = dict(re.findall(r"alphasat: variable (\d+) with score (\d+)", output))
    march_pos_lit_score_dict2 = {int(k):float(v) for k,v in march_pos_lit_score_dict2.items()}
    
    res = None
    if len(march_pos_lit_score_dict2) == 0:
        unsat_check = re.findall(r"c number of cubes (\d+), including (\d+) refuted leaf", output)
        if unsat_check[0][0] == unsat_check[0][1] in output:
            assert len(unsat_check) == 1
            res = 0
            print("Unsat!")
        elif "SATISFIABLE" in output:
            res = 1
            print("Sat!")
        else:
            print("Unknown result with empty dict!")
            print(output)
            print("-"*50)

    pos_keys2 = list(march_pos_lit_score_dict2.keys())
    neg_keys2 = [-k for k in pos_keys2]
    all_keys2 = pos_keys2 + neg_keys2

    for j in all_keys2:
        cnf = CNF(from_file=org_filename)

        print(k, j)
        cnf.append([k])
        cnf.append([j])
        all_results_nested[k][j] = {}

        cnf.to_file(filename)

        # ../PhysicsCheck/gen_cubes/march_cu/march_cu tmp.cnf -o tmp.cubes -d 1 -m 136
        result = subprocess.run(['../PhysicsCheck/gen_cubes/march_cu/march_cu', 
                                filename,
                                '-o',
                                'tmp.cubes', 
                                '-d', '1', '-m', str(edge_vars)], capture_output=True, text=True)

        output = result.stdout

        march_pos_lit_score_dict3 = dict(re.findall(r"alphasat: variable (\d+) with score (\d+)", output))
        march_pos_lit_score_dict3 = {int(k):float(v) for k,v in march_pos_lit_score_dict3.items()}

        res = None
        if len(march_pos_lit_score_dict3) == 0:
            unsat_check = re.findall(r"c number of cubes (\d+), including (\d+) refuted leaf", output)
            if unsat_check[0][0] == unsat_check[0][1] in output:
                assert len(unsat_check) == 1
                res = 0
                print("Unsat!")
            elif "SATISFIABLE" in output:
                res = 1
                print("Sat!")
            else:
                print("Unknown result with empty dict!")
                print(output)
                print("-"*50)

        if len(march_pos_lit_score_dict3) == 0:
            march_pos_lit_score_dict3[0] = 0.0
        maxv_key3 = max(march_pos_lit_score_dict3.items(), key=operator.itemgetter(1))[0]
        print(k, j, maxv_key3)

        total_score = march_pos_lit_score_dict[abs(k)] + march_pos_lit_score_dict2[abs(j)] + march_pos_lit_score_dict3[maxv_key3]
        all_results_nested[k][j][maxv_key3] = total_score

        if frozenset([k, j, maxv_key3]) in all_results:
            print("Duplicate!")
            print(k, j, maxv_key3)
            print("Existing val: ", all_results[frozenset([k, j, maxv_key3])])
            print("New val: ", total_score)
        else:
            all_results[frozenset([k, j, maxv_key3])] = total_score

    # save dict to pickle file
    with open("all_results.pickle", "wb") as f:
        pickle.dump(all_results, f)

    with open("all_results_nested.pickle", "wb") as f:
        pickle.dump(all_results_nested, f)
    
    # # load dict from pickle file
    # with open("all_results.pickle", "rb") as f:
    #     all_results = pickle.load(f)
    #     print(all_results)
    # 1/0

# print(all_results_nested)

# print(all_results)