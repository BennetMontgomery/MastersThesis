# IMPORTS
import libsumo
from agent import Agent

class ManualAgentObserver(Agent):
    def __init__(self, agentid, network, LANEUNITS=0.5, MAX_DECEL=7, MAX_ACCEL=4, verbose=False):
        super().__init__(agentid, network, LANEUNITS, MAX_DECEL, MAX_ACCEL, verbose)
        self.changed_route = False

    def select_action(self, time_step):
        # stabilize observer speed
        if libsumo.vehicle_getSpeed(self.agentid) < 2:
            self.change_speed(2, 1)
        elif libsumo.vehicle_getSpeed(self.agentid) > 2:
            self.change_speed(2, -1)

        # turn onto next roundabout section if on last roundabout section
        if libsumo.vehicle_getRoadID(self.agentid) == libsumo.vehicle_getRoute(self.agentid)[-1]:
            if libsumo.vehicle_getRoadID(self.agentid) == 'RE3': self.turn('RE1')
            if libsumo.vehicle_getRoadID(self.agentid) == 'RE2': self.turn('RE3')
            if libsumo.vehicle_getRoadID(self.agentid) == 'RE1': self.turn('RE2')