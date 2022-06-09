# IMPORTS 
import libsumo 
from agent import Agent


class ManualAgentSimple(Agent):
    def __init__(self, agentid, network, LANEUNITS=0.5, MAX_DECEL=7, MAX_ACCEL=4, verbose=False):
        super().__init__(agentid, network, LANEUNITS, MAX_DECEL, MAX_ACCEL, verbose)
        self.changed_left = False
        self.changed_right = False
        self.changing = False
        self.reached_3_speed = False
        self.reached_0_speed = False

    def select_action(self, timestep):
        if timestep == 1:
            libsumo.vehicle_setLaneChangeMode(self.agentid, 0)
            libsumo.vehicle_setSpeedMode(self.agentid, 0)

        if not self.reached_3_speed or (timestep > 23 and libsumo.vehicle_getSpeed(self.agentid) < 3):
            self.reached_3_speed = self.change_speed(3, 2)

        if libsumo.vehicle_getRoadID(self.agentid) == 'E2' and 'RE2' not in libsumo.vehicle_getRoute(self.agentid): 
            self.turn('RE2')

        if libsumo.vehicle_getRoadID(self.agentid) == 'RE2' and 'E-3' not in libsumo.vehicle_getRoute(self.agentid):
            self.turn('E-3')

        if timestep > 2 and not self.changed_left:
            self.changed_left = self.change_lane(1)
            self.changing = not self.changed_left

        if timestep > 9 and not self.changed_right:
            self.changed_right = self.change_lane(0)
            self.changing = not self.changed_right

        if not self.changing and libsumo.vehicle_getLateralLanePosition(self.agentid) != 0:
            self.lanewise_centre()

        if timestep >= 20 and not self.reached_0_speed:
            self.reached_0_speed = self.brake()

        if self.reached_0_speed:
            self.reached_3_speed = True