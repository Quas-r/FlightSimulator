class State(object):
    def __init__(self):
        self.enemy_position = []
        self.civilian_position = []
        self.enemy_rotation = []
        self.civilian_rotation = []
        self.enemy_velocity = 0
        self.civilian_velocity = 0
        self.enemy_g_force = 0
        self.civilian_g_force = 0
