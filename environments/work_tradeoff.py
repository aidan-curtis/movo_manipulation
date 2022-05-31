from environments.environment import Environment

from pybullet_planning.pybullet_tools.utils import create_box, TAN
import pybullet as p
  

class WorkTradeoff(Environment):
    def __init__(self):
        super(WorkTradeoff, self).__init__()

        self.start = [0, 0]
        self.goal = [0, 2] # TODO: Create separate class for configuration space

    def disconnect(self):
        try:
            p.disconnect()
        except:
            pass

    def setup(self):

        self.disconnect()
        
        p.connect(p.GUI)

        self.robot = self.setup_robot()
        self.room = self.create_closed_room(length=6, width=6)


        self.setup_occupancy_grid()
        self.setup_visibility_grid()    