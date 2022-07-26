import gym
from gym import spaces
from agents.agent import Agent
import numpy as np
import sumolib
import libsumo

class RoundaboutEnv(gym.Env):
    metadata = {"render_modes" : ["sumo-gui", "cli"]}

    def __init__(self, ego, render_mode: str = "cli"):
        assert render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.ego = ego # object reference to the ego agent

        # Observations are
        self.observation_space = spaces.Dict(
            {
                "number": spaces.Discrete(n=100, start=0),
                "speed": spaces.Box(low=-50, high=200, shape=(1,), dtype=np.float),
                "accel": spaces.Box(low=-7, high=4, shape=(1,), dtype=np.float),
                "laneid": spaces.Text(max_length=10),
                "lanepos": spaces.Box(low=0, high=100000, shape=(1,), dtype=np.float),
                "dti": spaces.Box(low=0, high=100000, shape=(1,), dtype=np.float)
            }
        )

        # Action space
        #   Index 0: Behaviour at time step
        #   Index 2: Speed change x0.5 at time step + 14
        self.action_space = spaces.MultiDiscrete([3, 22])



    ''' private method for collating observations for processing '''
    def _get_obs(self):
        # STATIC OBSERVATION VECTOR
        # number: visible vehicles
        # speed: current vehicle speed
        # accel: current vehicle acceleration
        # laneid: current lane the vehicle is in
        # lanepos: current longitudinal lane position
        # dti: distance to next intersection
        obs = {
            "number": len(self.ego.view),
            "speed": libsumo.vehicle_getSpeed(self.ego.agentid),
            "accel": libsumo.vehicle_getAcceleration(self.ego.agentid),
            "laneid": libsumo.vehicle_getLaneID(self.ego.agentid),
            "lanepos": libsumo.vehicle_getLanePosition(self.ego.agentid),
            "dti": libsumo.lane_getLength(libsumo.vehicle_getLaneID(self.ego.agentid))
                   - libsumo.vehicle_getLanePosition(self.ego.agentid)
        }

        # append vehicle tokens
        for vehicle in self.ego.view:
            vehicle_dict = {vehicle + str(self.ego.view.index(vehicle)): {
                "speed": libsumo.vehicle_getSpeed(vehicle),
                "laneid": libsumo.vehicle_getLaneID(vehicle),
                "lanepos": libsumo.vehicle_getLanePosition(vehicle)
            }}

            obs = obs | vehicle_dict

        # append adjacent lane data
        target_lanes = []
        for lane in range(libsumo.edge_getLaneNumber(libsumo.vehicle_getRoadID(self.ego.agentid))):
            if abs(lane - libsumo.vehicle_getLaneID(self.ego.agentid)) <= 1:
                target_lanes.append(lane)

        obs = obs | {"lanes": target_lanes}


        # append light data
        for light in libsumo.trafficlight_getIDList():
            if libsumo.vehicle_getLaneID(self.ego.agentid) is libsumo.trafficlight_getControlledLanes(light):
                light_dict = {light: libsumo.trafficlight_getRedYellowGreenState(light)}

                obs | light_dict

        return obs