import os
from collections import defaultdict
import json 

envs = ["simple_navigation", "simple_vision", "simple_namo", "complex", "subgoal_obstructed", "attachment_obstructed"]
algs = ["a_star", "namo", "vamp", "lamb", "snowplow"]
alg_name_map = {
    "A*":"a_star",
    "NAMO": "namo",
    "AC-NAMO":"snowplow" ,
    "VAMP": "vamp",
    "LaMB" : "lamb"
}

env_name_map = {
    "Simple Navigation": "simple_navigation",
    "Visibility":"simple_vision",
    "Movable Obstacles": "simple_namo",
    "Obstructed Visibility": "complex",
    "Occluding Obstacles": "subgoal_obstructed",
    "Obstructed Affordance": "attachment_obstructed"
}

filenames = []
results_dir = "/Users/aidancurtis/movo_manipulation/results/"
for file in os.listdir(results_dir):
    if file.endswith(".json"):
        filenames.append(os.path.join(results_dir, file))

results_dict = {}
for env in envs:
    results_dict[env] = defaultdict(list)
    for alg in algs:
        for filename in filenames:
            if("env={}".format(env) in filename and "algo={}".format(alg) in filename):
                with open(filename) as json_file:
                    data = json.load(json_file)
                    results_dict[env][alg].append(data)




print("\\toprule")
print("&{} \\\\ \\midrule".format("&".join(env_name_map.keys())))
for (r_alg, v_alg) in alg_name_map.items():
    env_results = []
    for (r_env, v_env) in env_name_map.items():
        num_success = sum([int(q["success"]) for q in results_dict[v_env][v_alg]])
        num_total = len(results_dict[v_env][v_alg])
        env_results.append("{}/{}".format(num_success, num_total))
    print("{} & {}\\\\".format(r_alg, " & ".join(env_results)))
print("\\bottomrule")