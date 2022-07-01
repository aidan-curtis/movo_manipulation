from planners.planner import Planner
from pybullet_planning.pybullet_tools.utils import wait_if_gui
#from planners.rrt import Graph, plot
import numpy as np

from utils.graph import Graph


class DoNothing(Planner):
    def __init__(self, env):
        super(DoNothing, self).__init__()

        self.env = env

    def get_plan(self):
        self.env.setup()
        G = Graph()
        G.initialize_full_graph(self.env)
        G.plot(self.env)

        wait_if_gui()




    def swept_volume(self, qs):