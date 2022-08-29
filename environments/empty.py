#from environments.environment import Environment
from environments.vamp_environment import Environment

from pybullet_planning.pybullet_tools.utils import LockRenderer
import pybullet as p
  

class Empty(Environment):
    def __init__(self):
        super(Empty, self).__init__()

        self.start = (0, 0, 0)
        self.goal = (0, 1, 0) # TODO: Create separate class for configuration space
        self.objects = []
        self.viewed_voxels = []

        # Properties represented as a list of width, length, height, mass
        self.objects_prop = dict()

    def disconnect(self):
        try:
            p.disconnect()
        except:
            pass

    def setup(self, G=None):

        self.disconnect()
        
        self.connect()

        self.robot = self.setup_robot()

        with LockRenderer():
            self.room = self.create_closed_room(length=6, width=6)
            self.setup_grids()

        self.objects += []
        self.static_objects = []
        self.centered_aabb = self.get_centered_aabb()
        self.centered_oobb = self.get_centered_oobb()