from planners.planner import Planner

class RandomSearch(Planner):
    def __init__(self):
        super(RandomSearch, self).__init__()

    def get_plan(self, environment):
        environment.setup()
        