import argparse
from pybullet_planning.pybullet_tools.utils import wait_if_gui
from planners.snowplow import Snowplow
from planners.a_star_search import AStarSearch
from planners.rrt import RRT
from planners.vamp import Vamp
from planners.lamb import Lamb
from planners.do_nothing import DoNothing
from environments.empty import Empty
from environments.complex import Complex
from environments.side_path import SidePath
from environments.single_movable import SingleMovable
from environments.simple_navigation import SimpleNavigation
from environments.attach_obstructed import AttObs
from environments.subgoal_obstructed import SubObs
from environments.simple_namo import SimpleNamo
from environments.simple_vision import SimpleVision
import pickle 
from datetime import datetime
import os


PLANNERS = {"snowplow": Snowplow,
            "a_star": AStarSearch,
            "rrt": RRT,
            "vamp": Vamp,
            "lamb": Lamb,
            "do_nothing": DoNothing}

ENVIRONMENTS = {"empty": Empty,
                "complex": Complex, 
                "side_path": SidePath,
                "single_movable": SingleMovable,
                "simple_navigation": SimpleNavigation,
                "attachment_obstructed": AttObs,
                "subgoal_obstructed": SubObs,
                "simple_namo": SimpleNamo,
                "simple_vision": SimpleVision}

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

    parser.add_argument(
        "-l",
        "--load",
        type=str,
        default=None,
        help="Data file indicating saved state"
    )

    parser.add_argument(
        "-d",
        "--debug",
        type=str,
        default="False",
        help="Whether to enter in debugging mode"
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
    plan = planner.get_plan(loadfile=args.load, debug=args.debug.lower() == "true")
    print(plan)
    wait_if_gui()
    statistics = env.validate_plan(plan)
    print(statistics)
    wait_if_gui()
    #write_results(args, statistics)