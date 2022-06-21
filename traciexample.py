# Bennet Montgomery 2022/23

## IMPORTS ##
import libsumo
import sumolib

import sys
sys.path.insert(1, './agents/')
from agents.manual_agent_simple import ManualAgentSimple
from agents.manual_agent_observer import ManualAgentObserver

## CONSTANTS ##
MAX_DECEL=7 # see brake()
MAX_ACCEL=4 # see change_speed()
NETWORK=sumolib.net.readNet('./Sumoconfgs/simple.net.xml')
LANEWIDTH=3
LANEUNITS=1.0

libsumo.start(["sumo-gui", "-c", "./Sumoconfgs/simple.sumocfg", "--lateral-resolution=3.0"])

# step to 1
libsumo.simulationStep()
# instantiate agent
default = ManualAgentSimple('agent', NETWORK, LANEUNITS, MAX_DECEL, MAX_ACCEL)
observer = ManualAgentObserver('observer', NETWORK, LANEUNITS, MAX_DECEL, MAX_ACCEL)
agent_left = False

for time_step in range(1, 75):

    try:
        default.select_action(libsumo.simulation_getTime())
        observer.select_action(libsumo.simulation_getTime())

        libsumo.simulationStep()
    except libsumo.libsumo.TraCIException:
        if not agent_left:
            print('agent has left')
            agent_left = True

libsumo.close()
