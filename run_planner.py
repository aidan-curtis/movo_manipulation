import argparse
from planners.snowplow import Snowplow
from planners.a_star_search import AStarSearch
from planners.rrt import RRT
from planners.do_nothing import DoNothing
from environments.empty import Empty
from environments.complex import Complex
from environments.work_tradeoff import WorkTradeoff
from environments.side_path import SidePath
from environments.single_movable import SingleMovable
from environments.double_movable import DoubleMovable
import pickle 
from datetime import datetime
import os

PLANNERS = {"snowplow": Snowplow,
            "a_star": AStarSearch,
            "rrt": RRT,
            "do_nothing": DoNothing}

ENVIRONMENTS = {"empty": Empty,
                "complex": Complex, 
                "work_tradeoff": WorkTradeoff,
                "side_path": SidePath,
                "single_movable": SingleMovable,
                "double_movable": DoubleMovable}
RESULTS_DIR = "./results"

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-a",
        "--algo",
        default="rrt",
        type=str,
        help="Planning algorithm to run",
        choices=list(PLANNERS.keys())
    )

    parser.add_argument(
        "-e",
        "--env",
        type=str,
        default="single_movable",
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

    env = ENVIRONMENTS[args.env]()

    planner = PLANNERS[args.algo](env)
    plan = planner.get_plan()
    statistics = env.validate_plan(plan)

    #write_results(args, statistics)