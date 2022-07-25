import gym
from gym import spaces
import numpy as np
import sumolib
import libsumo

class RoundaboutEnv(gym.Env):
    metadata = {"render_modes" : ["sumo-gui", "cli"]}

    def __init__(self, render_mode: str = "cli", ego: str = "proposed_agent"):
        assert render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.ego = ego # the identity of the ego agent

        # Observations are
        self.observation_space = spaces.Dict(

        )

        # Action space
        #   Index 0: Behaviour at time step
        #   Index 2: Speed change x0.5 at time step + 14
        self.action_space = spaces.MultiDiscrete([3, 28])



    ''' private method for collating observations for processing '''
    def _get_obs(self):
        return {"number": libsumo.vehicle_getIDCount(), ""}

