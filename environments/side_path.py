from environments.environment import Environment

from pybullet_planning.pybullet_tools.utils import create_box, TAN, BROWN, set_pose, Pose, Point
import pybullet as p
  

class SidePath(Environment):
    def __init__(self):
        super(SidePath, self).__init__()

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
        self.room = self.create_closed_room(length=6, width=10, center = [3,2])


        blocking_box = create_box(3,4,1, mass=1, color=BROWN)
        set_pose(blocking_box,
                Pose(point=Point(
                        x=3,
                        y=1,
                        z=1/2,
                    )
                )
            )



        self.setup_occupancy_grid()
        self.setup_visibility_grid()