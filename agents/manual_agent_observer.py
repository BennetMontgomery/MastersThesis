# IMPORTS
import libsumo
from agent import Agent

class ManualAgentObserver(Agent):
    def __init__(self, agentid, network, LANEUNITS=0.5, MAX_DECEL=7, MAX_ACCEL=4, verbose=False, VIEW_DISTANCE=30):
        super().__init__(agentid, network, LANEUNITS, MAX_DECEL, MAX_ACCEL, verbose, VIEW_DISTANCE)
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

        # print observations
        self.observe_surroundings()

    def observe_surroundings(self, target_vehicle=None):
        # ensure surroundings are up to date
        self.refresh_view()

        # list stats of each observed vehicle
        for vehicle in self.view:
            if target_vehicle is None or target_vehicle == vehicle:
                signals_active = libsumo.vehicle_getSignals(vehicle)
                print("Observer sees " + vehicle + " at time step " + str(libsumo.simulation_getTime()))
                print("\ton edge " + libsumo.vehicle_getRoadID(vehicle) + " moving at speed " + str(libsumo.vehicle_getSpeed(vehicle)))
                print("\tvehicle is in lane " + str(libsumo.vehicle_getLaneID(vehicle)))
                print("\tvehicle signals active: " + (" brake " if (signals_active&(1<<3))>>3 else "")
                      + (" right " if (signals_active&1) else "")
                      + (" left " if (signals_active&(1<<1))>>1 else ""))

                # highlight vehicle in gui
                libsumo.vehicle_setColor(vehicle, (0, 255, 0, 255))

        # unhighlight vehicles out of view
        for vehicle in libsumo.vehicle_getIDList():
            if vehicle not in self.view:
                libsumo.vehicle_setColor(vehicle, libsumo.vehicletype_getColor(libsumo.vehicle_getTypeID(vehicle)))