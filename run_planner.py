import argparse
from pybullet_planning.pybullet_tools.utils import wait_if_gui, draw_aabb
from planners.snowplow import Snowplow
from planners.a_star_search import AStarSearch
from planners.rrt import RRT
from planners.vamp import Vamp
from planners.lamb import Lamb
from planners.namo import Namo
from planners.rotate import Rotate
from planners.do_nothing import DoNothing
from environments.empty import Empty
from environments.complex import Complex
from environments.simple_navigation import SimpleNavigation
from environments.attach_obstructed import AttObs
from environments.subgoal_obstructed import SubObs
from environments.simple_namo import SimpleNamo
from environments.simple_vision import SimpleVision
from environments.real_world import RealWorld
import pickle
from datetime import datetime
import os
import time
import random
import numpy as np

PLANNERS = {"snowplow": Snowplow,
            "a_star": AStarSearch,
            "namo": Namo,
            "rrt": RRT,
            "vamp": Vamp,
            "lamb": Lamb,
            "do_nothing": DoNothing,
            "rotate": Rotate}

ENVIRONMENTS = {"empty": Empty,
                "complex": Complex, 
                "simple_navigation": SimpleNavigation,
                "attachment_obstructed": AttObs,
                "subgoal_obstructed": SubObs,
                "simple_namo": SimpleNamo,
                "simple_vision": SimpleVision,
                "real_world": RealWorld}


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
        "-ov",
        "--only_validate",
        type=str,
        default=None,
        help="Filename of the plan to run validation on"
    )

    parser.add_argument(
        "-l",
        "--load",
        type=str,
        default=None,
        help="Data file indicating saved state"
    )

    parser.add_argument(
        "-s",
        "--save_dir",
        type=str,
        default="./results",
        help="Place to save statistics"
    )

    parser.add_argument(
        "-d",
        "--debug",
        type=str,
        default="False",
        help="Whether to enter in debugging mode"
    )

    parser.add_argument(
        "-seed",
        "--seed",
        type=int,
        default=0,
        help="seed"
    )


    args = parser.parse_args()
    return args

def write_results(args, statistics, save_dir):
    fn = "algo={}_env={}_seed={}.pkl".format(args.algo, args.env, args.seed)
    results_fn = os.path.join(save_dir, fn)
    with open(results_fn, 'wb') as handle:
        pickle.dump(statistics, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__=="__main__":
    args = get_args()

    print("=================")
    print(args)

    random.seed(args.seed)
    np.random.seed(args.seed)
    env = ENVIRONMENTS[args.env](vis=args.vis)
    planner = PLANNERS[args.algo](env)
    start_time = time.time()
    if(args.only_validate is None):
        plan = planner.get_plan(loadfile=args.load, debug=args.debug.lower() == "true")
    else:
        with open(args.only_validate, 'rb') as handle:
            data = pickle.load(handle)
            plan = data["plan"]

    plan_time = time.time()-start_time
    print(plan)
    wait_if_gui()
    statistics = env.validate_plan(plan)

    print(statistics)
    if(args.vis):
        for q, att in statistics[1]:
            draw_aabb(env.aabb_from_q(q))
            if att is not None:
                draw_aabb(env.movable_object_oobb_from_q(att[0], q, att[1]).aabb)
            env.move_robot(q, env.joints, att)
            wait_if_gui()
        wait_if_gui()
        
    results_dict = {"success": statistics[0],
                    "collisions":len(statistics[1]), 
                    "plan": plan,
                    "plan_time":plan_time}

    write_results(args, results_dict, args.save_dir)
    time.sleep(5)
