from planners.planner import Planner
from pybullet_planning.pybullet_tools.utils import wait_if_gui
from planners.rrt import Graph, plot
import numpy as np


class DoNothing(Planner):
    def __init__(self, env):
        super(DoNothing, self).__init__()

        self.env = env

    def get_plan(self):
        self.env.setup()
        G = Graph(self.env.start, self.env.goal)
        plot(G, self.env)

        wait_if_gui()





