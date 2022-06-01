import argparse
from planners.random_search import RandomSearch
from planners.a_star_search import AStarSearch
from planners.rrt import RRT
from environments.empty import Empty
from environments.complex import Complex
from environments.work_tradeoff import WorkTradeoff
from environments.side_path import SidePath
import pickle 
from datetime import datetime
import os
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d


PLANNERS = {"random_search": RandomSearch,
            "a_star": AStarSearch,
            "rrt": RRT}
ENVIRONMENTS = {"empty": Empty,
                "complex": Complex, 
                "work_tradeoff": WorkTradeoff,
                "side_path": SidePath}
RESULTS_DIR = "./results"

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-a",
        "--algo",
        default="random_search",
        type=str,
        help="Planning algorithm to run",
        choices=list(PLANNERS.keys())
    )

    parser.add_argument(
        "-e",
        "--env",
        type=str,
        default="empty",
        help="Environment to run the planner in",
        choices=list(ENVIRONMENTS.keys())
    )

    parser.add_argument(
        "-v",
        "--vis",
        action="store_true"
    )
    args = parser.parse_args()
    return args

def write_results(args, statistics):
    now = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    fn = "algo={}_env={}_t={}".format(args.algo, args.env, now)
    results_fn = os.path.join(RESULTS_DIR, fn)
    with open(results_fn, 'wb') as handle:
        pickle.dump(statistics, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__=="__main__":
    args = get_args()

    planner = PLANNERS[args.algo]()
    env = ENVIRONMENTS[args.env]()

    plan = planner.get_plan(env)
    statistics = env.validate_plan(plan)

    #write_results(args, statistics)