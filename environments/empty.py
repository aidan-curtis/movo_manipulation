from environments.environment import Environment

from pybullet_planning.pybullet_tools.utils import LockRenderer
import pybullet as p
  

class Empty(Environment):
    def __init__(self):
        super(Empty, self).__init__()

        self.start = (0, 0, 0)
        self.goal = (0, 1, 0) # TODO: Create separate class for configuration space

    def disconnect(self):
        try:
            p.disconnect()
        except:
            pass

    def setup(self):

        self.disconnect()
        
        p.connect(p.GUI)

        self.robot = self.setup_robot()

        with LockRenderer():
            self.room = self.create_closed_room(length=6, width=6)


            self.setup_grids()