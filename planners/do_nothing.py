from planners.planner import Planner
from pybullet_planning.pybullet_tools.utils import wait_if_gui
from utils.graph import Graph
import numpy as np

GRID_RESOLUTION = 0.2


class DoNothing(Planner):
    def __init__(self, env):
        super(DoNothing, self).__init__()

        self.env = env

    def get_plan(self, loadfile=None, debug=False):
        self.env.setup()
        G = Graph()
        G.initialize_full_graph(self.env, [GRID_RESOLUTION, GRID_RESOLUTION, np.pi/8])
        self.env.restrict_configuration(G)
        G.plot(self.env)
        wait_if_gui()
