# Bennet Montgomery 2022/23

# IMPORTS #
import libsumo
from abc import ABC, abstractmethod
from math import sqrt


class Agent(ABC):
    def __init__(self, agentid, network, LANEUNITS=0.5, MAX_DECEL=7, MAX_ACCEL=4, verbose=False, VIEW_DISTANCE=0):
        # Class constants
        # Default vehicles have a max deceleration of 7 and acceleration of 4
        self.agentid = agentid
        self.NETWORK = network
        self.LANEUNITS = 0.5
        self.MAX_DECEL = 7
        self.MAX_ACCEL = 4
        self.verbose = verbose
        self.VIEW_DISTANCE = VIEW_DISTANCE

        # vehicles in sight
        self.view = []

        # async update variables
        self.lane = None

    # ALL AGENTS MUST HAVE A DEFINED BEHAVIOUR GIVEN THE CURRENT TIMESTEP
    @abstractmethod
    def select_action(self, time_step):
        pass

    def brake(self):
        # a standard car brakes at -7 m/s2 (Australian government):
        # https://www.qld.gov.au/transport/safety/road-safety/driving-safely/stopping-distances
        if self.verbose:
            print("Braking at time step " + str(libsumo.simulation_getTime()))

        if libsumo.vehicle_getSpeed(self.agentid) > self.MAX_DECEL and libsumo.vehicle_getSpeed(self.agentid) > 0:
            libsumo.vehicle_setSpeed(self.agentid, libsumo.vehicle_getSpeed(self.agentid) - self.MAX_DECEL)
            return False # action incomplete
        else:
            libsumo.vehicle_setSpeed(self.agentid, 0)
            return True # action complete

    def change_speed(self, new_speed, acceleration):
        # most sedans can comfortably accelerate 4 m/s2 https://hypertextbook.com/facts/2001/MeredithBarricella.shtml
        if self.verbose:
            print("Changing speed from " + str(libsumo.vehicle_getSpeed(self.agentid)) + " to " + str(new_speed) + " at timestep " + str(libsumo.simulation_getTime()))

        # cap acceleration
        acceleration = self.MAX_ACCEL if acceleration > self.MAX_ACCEL else acceleration
        acceleration = -self.MAX_DECEL if acceleration < -self.MAX_DECEL else acceleration

        if libsumo.vehicle_getSpeed(self.agentid) < new_speed:
            if abs(new_speed - libsumo.vehicle_getSpeed(self.agentid)) > acceleration:
                update = libsumo.vehicle_getSpeed(self.agentid) + acceleration
                libsumo.vehicle_setSpeed(self.agentid, update)
                return False # action incomplete
            else:
                libsumo.vehicle_setSpeed(self.agentid, new_speed)
                return True # action complete

    def turn(self, new_edge):
        if self.verbose:
            print("adding turn onto " + new_edge + " at time step " + str(libsumo.simulation_getTime()))

        outgoing = self.NETWORK.getEdge(libsumo.vehicle_getRoadID(self.agentid)).getOutgoing()
        outgoing_edges = [outgoing[list(outgoing.keys())[i]][0]._to.getID() for i in range(len(outgoing))]

        if new_edge in outgoing_edges:
            libsumo.vehicle_setRoute(self.agentid, [libsumo.vehicle_getRoadID(self.agentid), new_edge])

    def lanewise_centre(self):
        if self.verbose:
            print("centring vehicle at timestep " + str(libsumo.simulation_getTime()) + " in lane " + str(libsumo.vehicle_getLaneID(self.agentid)))

        if abs(libsumo.vehicle_getLateralLanePosition(self.agentid)) <= self.LANEUNITS:
            libsumo.vehicle_changeSublane(self.agentid, -libsumo.vehicle_getLateralLanePosition(self.agentid))
            return True

        if libsumo.vehicle_getLateralLanePosition(self.agentid) > self.LANEUNITS:
            libsumo.vehicle_changeSublane(self.agentid, -self.LANEUNITS)
            return False
        elif libsumo.vehicle_getLateralLanePosition(self.agentid) < -self.LANEUNITS:
            libsumo.vehicle_changeSublane(self.agentid, self.LANEUNITS)
            return False

    def change_lane(self, new_lane, duration=1000):
        if self.verbose:
            print("changing lane at timestep " + str(libsumo.simulation_getTime()))
            print("Lanechangemode: " + str(libsumo.vehicle_getLaneChangeMode(self.agentid)))

        libsumo.vehicle_changeLane(self.agentid, new_lane, duration=duration)

        if self.lane != libsumo.vehicle_getLaneID(self.agentid):
            self.lane = libsumo.vehicle_getLaneID(self.agentid)
            return True
        else:
            return False

    def flip_verbose(self):
        self.verbose = not self.verbose

    # American English friendly aliasing
    def lanewise_center(self):
        self.lanewise_centre()

    def refresh_view(self):
        coords = libsumo.vehicle_getPosition(self.agentid)

        self.view = []
        tmp_view = []

        # list all vehicles within VIEW_DISTANCE
        for vehicle in [i for i in libsumo.vehicle_getIDList() if i != self.agentid]:
            vehicle_coords = libsumo.vehicle_getPosition(vehicle)
            dist = sqrt((vehicle_coords[0]-coords[0])**2 + (vehicle_coords[1]-coords[1])**2)
            if dist < self.VIEW_DISTANCE:
                tmp_view.append((vehicle, dist))

        # sort vehicles by distance
        tmp_view = sorted(tmp_view, key=lambda veh: veh[1])

        # remove vehicles blocked by other vehicles
        # a vehicle counts as blocking the view if drawing a thread between the source and the target intersects a
        # circle at the centre of the blocking vehicle with radius equal to half a vehicle width

        # closest vehicle cannot be blocked
        if len(tmp_view) > 0:
            surveyed_vehicles = [tmp_view.pop(0)[0]]
            self.view.append(surveyed_vehicles[0])

        while len(tmp_view) > 0:
            vehicle = tmp_view.pop(0)
            veh_coords = libsumo.vehicle_getPosition(vehicle[0])
            # determine blockage by closer vehicles
            viewable = True
            for surv in surveyed_vehicles:
                surv_coords = libsumo.vehicle_getPosition(surv)
                # calculate Determinant relative to potential blockage
                D = (coords[0]-surv_coords[0])*(veh_coords[1]-surv_coords[1]) - (veh_coords[0]-surv_coords[0])*(coords[1]-surv_coords[1])
                # calculate intersection. If r*r*d*d - D*D > 0, there is an intersection
                if ((libsumo.vehicle_getWidth(surv)/2)**2) * (vehicle[1]**2) - (D**2) > 0:
                    # calculate point of intersect. If it falls between the two vehicles, viewage is blocked
                    dy = veh_coords[1]-coords[1]
                    dx = veh_coords[0]-coords[0]
                    if dy < 0:
                        x = (D*dy + dx*sqrt((libsumo.vehicle_getWidth(surv)/2)**2) * (vehicle[1]**2) - (D**2))/((libsumo.vehicle_getWidth(surv)/2)**2)
                    else:
                        x = (D*dy - dx*sqrt((libsumo.vehicle_getWidth(surv)/2)**2) * (vehicle[1]**2) - (D**2))/((libsumo.vehicle_getWidth(surv)/2)**2)

                    y = (-D*dx + abs(dy) * sqrt((libsumo.vehicle_getWidth(surv)/2)**2) * (vehicle[1]**2) - (D**2))/((libsumo.vehicle_getWidth(surv)/2)**2)

                    if (dx < max(coords[0], veh_coords[0]) and dx > min(coords[0], veh_coords[0])) and (dy < max(coords[1], veh_coords[1]) and dy < min(coords[1], veh_coords[1])):
                        viewable = False
                        print("vehicle " + vehicle[0] + " is blocked from view by vehicle " + surv)
                        break

            if viewable:
                self.view.append(vehicle[0])

            surveyed_vehicles.append(vehicle[0])

        if self.verbose:
            print(self.agentid + " sees " + str(self.view) + " at timestep " + str(libsumo.simulation_getTime()))

    def set_network(self, new_net):
        self.NETWORK = new_net