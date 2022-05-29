# IMPORTS 
import libsumo 
from agent import Agent


class ManualAgentSimple(Agent):
    def __init__(self, agentid, network, LANEUNITS=0.5, MAX_DECEL=7, MAX_ACCEL=4, verbose=False):
        super().__init__(agentid, network, LANEUNITS, MAX_DECEL, MAX_ACCEL, verbose)
        self.changed_left = False
        self.changed_right = False

    def select_action(self, timestep, state=None):
        if timestep == 1:
            libsumo.vehicle_setLaneChangeMode(self.agentid, 1)
            self.change_speed(3, 2)

        if libsumo.vehicle_getRoadID(self.agentid) == 'E2' and 'RE2' not in libsumo.vehicle_getRoute(self.agentid): 
            self.turn('RE2')

        if libsumo.vehicle_getRoadID(self.agentid) == 'RE2' and 'E-3' not in libsumo.vehicle_getRoute(self.agentid):
            self.turn('E-3')

        if timestep > 2 and not self.changed_left:
            self.change_lane(1)
            self.changed_left = True

        if timestep > 9 and not self.changed_right:
            self.change_lane(0)
            self.changed_right = True


        if timestep == 18:
            self.change_speed(0.1, -1)

        if timestep == 22:
            self.change_speed(3, 1)
