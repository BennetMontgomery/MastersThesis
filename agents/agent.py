# Bennet Montgomery 2022/23

# IMPORTS #
import libsumo
from abc import ABC, abstractmethod


class Agent(ABC):
    def __init__(self, agentid, network, LANEUNITS=0.5, MAX_DECEL=7, MAX_ACCEL=4, verbose=False):
        # Class constants
        # Default vehicles have a max deceleration of 7 and acceleration of 4
        self.agentid = agentid
        self.NETWORK = network
        self.LANEUNITS = 0.5
        self.MAX_DECEL = 7
        self.MAX_ACCEL = 4
        self.verbose = verbose

    # ALL AGENTS MUST HAVE A DEFINED BEHAVIOUR GIVEN THE CURRENT TIMESTEP
    @abstractmethod
    def select_action(self, time_step, state=None):
        pass

    def brake(self):
        # a standard car brakes at -7 m/s2 (Australian government):
        # https://www.qld.gov.au/transport/safety/road-safety/driving-safely/stopping-distances
        if self.verbose:
            print("Braking at time step " + str(libsumo.simulation_getTime()))

        if libsumo.vehicle_getSpeed(self.agentid)/self.MAX_DECEL < 1:
            libsumo.vehicle_setSpeed(self.agentid, 0)
        else:
            for time_step in range(int(libsumo.vehicle_getSpeed(self.agentid)/self.MAX_DECEL) + 1):
                if libsumo.vehicle_getSpeed(self.agentid) > self.MAX_DECEL:
                    libsumo.vehicle_setSpeed(self.agentid, libsumo.vehicle_getSpeed(self.agentid) - self.MAX_DECEL)
                else:
                    libsumo.vehicle_setSpeed(self.agentid, 0)

                libsumo.simulationStep()

    def change_speed(self, new_speed, acceleration):
        # most sedans can comfortably accelerate 4 m/s2 https://hypertextbook.com/facts/2001/MeredithBarricella.shtml
        if self.verbose:
            print("Changing speed from " + str(libsumo.vehicle_getSpeed(self.agentid)) + " to " + str(new_speed) + " at timestep " + str(libsumo.simulation_getTime()))

        # cap acceleration
        acceleration = self.MAX_ACCEL if acceleration > self.MAX_ACCEL else acceleration
        acceleration = -self.MAX_DECEL if acceleration < -self.MAX_DECEL else acceleration

        # increase speed if new_speed is higher
        while libsumo.vehicle_getSpeed(self.agentid) < new_speed:
            if new_speed - libsumo.vehicle_getSpeed(self.agentid) > acceleration:
                update = libsumo.vehicle_getSpeed(self.agentid) + acceleration
                libsumo.vehicle_setSpeed(self.agentid, update)
                libsumo.simulationStep()
            else:
                libsumo.vehicle_setSpeed(self.agentid, new_speed)
                libsumo.simulationStep()

        # decrease speed if new_speed is lower
        while libsumo.vehicle_getSpeed(self.agentid) > new_speed:
            if libsumo.vehicle_getSpeed(self.agentid) - new_speed > acceleration:
                libsumo.vehicle_setSpeed(self.agentid, libsumo.vehicle_getSpeed(self.agentid) + acceleration)
                libsumo.simulationStep()
            else:
                libsumo.vehicle_setSpeed(self.agentid, new_speed)
                libsumo.simulationStep()

    def turn(self, new_edge):
        if self.verbose:
            print("adding turn onto " + new_edge + " at time step " + str(libsumo.simulation_getTime())) 

        outgoing = self.NETWORK.getEdge(libsumo.vehicle_getRoadID(self.agentid)).getOutgoing()
        outgoing_edges = [outgoing[list(outgoing.keys())[i]][0]._to.getID() for i in range(len(outgoing))]

        if new_edge in outgoing_edges:
            libsumo.vehicle_setRoute(self.agentid, [libsumo.vehicle_getRoadID(self.agentid), new_edge])

    def change_lane(self, new_lane, duration=100):
        if self.verbose:
            print("changing lane at timestep " + str(libsumo.simulation_getTime()))
            print("Lanechangemode: " + str(libsumo.vehicle_getLaneChangeMode(self.agentid)))

        libsumo.vehicle_changeLane(self.agentid, new_lane, duration=1000)
        libsumo.simulationStep()
        libsumo.simulationStep()

        if new_lane > libsumo.vehicle_getLaneIndex(self.agentid):
            while libsumo.vehicle_getLateralLanePosition(self.agentid) < -self.LANEUNITS:
                libsumo.vehicle_changeSublane(self.agentid,self.LANEUNITS)
                libsumo.simulationStep()

            libsumo.vehicle_changeSublane(self.agentid, -libsumo.vehicle_getLateralLanePosition(self.agentid))
            libsumo.simulationStep()
        else:
            while libsumo.vehicle_getLateralLanePosition(self.agentid) > self.LANEUNITS:
                libsumo.vehicle_changeSublane(self.agentid,-self.LANEUNITS)
                libsumo.simulationStep()

            libsumo.vehicle_changeSublane(self.agentid, -libsumo.vehicle_getLateralLanePosition(self.agentid))
            libsumo.simulationStep()

        libsumo.vehicle_changeSublane(self.agentid, 0)

    def flip_verbose(self):
        self.verbose = not self.verbose
