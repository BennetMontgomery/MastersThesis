# Bennet Montgomery 2022/23

## IMPORTS ##
import libsumo
import sumolib

## CONSTANTS ##
MAX_DECEL=7 # see brake()
MAX_ACCEL=4 # see change_speed()
NETWORK=sumolib.net.readNet('./Sumoconfgs/simple.net.xml')
LANEWIDTH=3
LANEUNITS=1.0


def brake():
    # a standard car brakes at -7 m/s2 (Australian government): https://www.qld.gov.au/transport/safety/road-safety/driving-safely/stopping-distances
    print("Braking at time step " + str(libsumo.simulation_getTime()))

    if libsumo.vehicle_getSpeed('agent')/MAX_DECEL < 1:
        libsumo.vehicle_setSpeed('agent', 0)
    else:
        for time_step in range(int(libsumo.vehicle_getSpeed('agent')/MAX_DECEL) + 1):
            if libsumo.vehicle_getSpeed('agent') > MAX_DECEL:
                libsumo.vehicle_setSpeed('agent', libsumo.vehicle_getSpeed('agent') - MAX_DECEL)
            else:
                libsumo.vehicle_setSpeed('agent', 0)

            libsumo.simulationStep()


def change_speed(new_speed, acceleration):
    # most sedans can comfortably accelerate 4 m/s2 https://hypertextbook.com/facts/2001/MeredithBarricella.shtml
    print("Changing speed from " + str(libsumo.vehicle_getSpeed('agent')) + " to " + str(new_speed) + " at timestep " + str(libsumo.simulation_getTime()))

    # cap acceleration
    acceleration = MAX_ACCEL if acceleration > MAX_ACCEL else acceleration
    acceleration = -MAX_DECEL if acceleration < -MAX_DECEL else acceleration

    # increase speed if new_speed is higher
    while libsumo.vehicle_getSpeed('agent') < new_speed:
        if new_speed - libsumo.vehicle_getSpeed('agent') > acceleration:
            update = libsumo.vehicle_getSpeed('agent') + acceleration
            libsumo.vehicle_setSpeed('agent', update)
            libsumo.simulationStep()
        else:
            libsumo.vehicle_setSpeed('agent', new_speed)
            libsumo.simulationStep()

    # decrease speed if new_speed is lower
    while libsumo.vehicle_getSpeed('agent') > new_speed:
        if libsumo.vehicle_getSpeed('agent') - new_speed > acceleration:
            libsumo.vehicle_setSpeed('agent', libsumo.vehicle_getSpeed('agent') + acceleration)
            libsumo.simulationStep()
        else:
            libsumo.vehicle_setSpeed('agent', new_speed)
            libsumo.simulationStep()


def turn(new_edge):
    outgoing = NETWORK.getEdge(libsumo.vehicle_getRoadID('agent')).getOutgoing()
    outgoing_edges = [outgoing[list(outgoing.keys())[i]][0]._to.getID() for i in range(len(outgoing))]

    if new_edge in outgoing_edges:
        libsumo.vehicle_setRoute('agent', [libsumo.vehicle_getRoadID('agent'), new_edge])


def change_lane(new_lane, duration=100):
    print("changing lane at timestep " + str(libsumo.simulation_getTime()))
    print("Lanechangemode: " + str(libsumo.vehicle_getLaneChangeMode('agent')))

    libsumo.vehicle_changeLane('agent', new_lane, duration)
    libsumo.simulationStep()
    libsumo.simulationStep()

    if new_lane > libsumo.vehicle_getLaneIndex('agent'):
        while libsumo.vehicle_getLateralLanePosition('agent') < -LANEUNITS:
            libsumo.vehicle_changeSublane('agent',LANEUNITS)
            libsumo.simulationStep()

        libsumo.vehicle_changeSublane('agent', -libsumo.vehicle_getLateralLanePosition('agent'))
        libsumo.simulationStep()
    else:
        while libsumo.vehicle_getLateralLanePosition('agent') > LANEUNITS:
            libsumo.vehicle_changeSublane('agent',-LANEUNITS)
            libsumo.simulationStep()

        libsumo.vehicle_changeSublane('agent', -libsumo.vehicle_getLateralLanePosition('agent'))
        libsumo.simulationStep()

    libsumo.vehicle_changeSublane('agent', 0)



def default_agent():
    libsumo.start(["sumo-gui", "-c", "./Sumoconfgs/simple.sumocfg", "--lateral-resolution=3.0"])

    libsumo.simulationStep() # step to time 0

    print(libsumo.vehicle_getIDList())

    libsumo.vehicle_setLaneChangeMode('agent', 1)

    # accelerate to 50km/h
    change_speed(3, 2)

    # plan trip
    turn('RE2')

    print(libsumo.vehicle_getRoute('agent'))
    change_lane(1)

    libsumo.simulationStep(libsumo.simulation_getTime() + 3)

    change_lane(0)

    libsumo.simulationStep(18)

    # rolling brake
    change_speed(0.1, -1)

    libsumo.simulationStep()

    # restore speed
    change_speed(3, 1)

    libsumo.simulationStep(libsumo.simulation_getTime() + 2)

    turn('E-3')

    libsumo.simulationStep(libsumo.simulation_getTime() + 28)

    libsumo.close()


default_agent()