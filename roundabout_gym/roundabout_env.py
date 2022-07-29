import gym
from gym import spaces
from agents.agent import Agent
import libsumo
import os
import random

# Constants
CONFIGS_PATH="../Sumoconfgs/"
EGO_TYPE = ['"agenttype"', '"4.0"', '"7.0"', '"5"', '"30"', '"red"']
NPC_TYPE = ['"npctype"', '"0.8"', '"5"', '"14"']
MIN_NPCS = 3
MAX_NPCS = 6
SPAWN_TIME_RANGE = 16


class RoundaboutEnv(gym.Env):
    metadata = {"render_modes" : ["sumo-gui", "cli"]}

    def __init__(self, ego, render_mode: str = "cli"):
        assert render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.ego = ego # object reference to the ego agent

        # Observations are
        # self.observation_space = spaces.Dict(
        #     {
        #         "number": spaces.Discrete(n=100, start=0),
        #         "speed": spaces.Box(low=-50, high=200, shape=(1,), dtype=np.float),
        #         "accel": spaces.Box(low=-7, high=4, shape=(1,), dtype=np.float),
        #         "laneid": spaces.Text(max_length=10),
        #         "lanepos": spaces.Box(low=0, high=100000, shape=(1,), dtype=np.float),
        #         "dti": spaces.Box(low=0, high=100000, shape=(1,), dtype=np.float)
        #     }
        # )

        # Action space
        #   Index 0: Behaviour at time step
        #   Index 2: Speed change x0.5 at time step + 14
        self.action_space = spaces.MultiDiscrete([3, 22])

        self.npc_depart_times = []



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
                "laneid": libsumo.vehicle_getLaneIndex(vehicle),
                "lanepos": libsumo.vehicle_getLanePosition(vehicle)
            }}

            obs = obs | vehicle_dict

        # append adjacent lane data
        target_lanes = []
        for lane in range(libsumo.edge_getLaneNumber(libsumo.vehicle_getRoadID(self.ego.agentid))):
            if abs(lane - libsumo.vehicle_getLaneIndex(self.ego.agentid)) <= 1:
                target_lanes.append(lane)

        obs = obs | {"lanes": target_lanes}


        # append light data
        for light in libsumo.trafficlight_getIDList():
            if libsumo.vehicle_getLaneID(self.ego.agentid) is libsumo.trafficlight_getControlledLanes(light):
                light_dict = {light: libsumo.trafficlight_getRedYellowGreenState(light)}

                obs | light_dict

        return obs

    def reset(self, seed=None, return_info=False, options=None):
        # Seed random generator
        random.seed()
        super().reset(seed=seed)

        ''' Open configuration files and select roundabout model '''
        sumoconfigs = os.listdir("{configs}".format(configs=CONFIGS_PATH))
        networks = [entry for entry in sumoconfigs if entry.endswith(".net.xml")]

        # select random network
        network = random.choice(networks)
        route = "{net}rou.xml".format(net=network[:-7])

        # place ego agent at lane 0 in a random edge at timestep 0
        file = open(CONFIGS_PATH + "{net}".format(net=network))
        edges = []
        for line in file:
            line = line.strip()
            if line is not None and "edge id=" in line and "function=\"internal\"" not in line:
                reached_edges = True
                edges.append(line.split("edge id=")[1].split("\"")[1].split("\"")[0])

        file.close()

        # always select a starting edge pointing towards a roundabout
        start_edge = random.choice([edge for edge in edges if "-" not in edge and "R" not in edge])
        # select a finishing edge pointing away from the roundabout
        finish_edge = random.choice([edge for edge in edges if "-" in edge])
        ego_type_line = '<vType id={ID} accel={ACCEL} emergencyDecel={EDECEL} sigma="0" length={LENGTH} maxspeed={MAX} color={COLOUR}/>'.format(
            ID = EGO_TYPE[0], ACCEL = EGO_TYPE[1], EDECEL = EGO_TYPE[2], LENGTH = EGO_TYPE[3], MAX = EGO_TYPE[4], COLOUR = EGO_TYPE[5]
        )
        ego_route_line = '<vehicle id="{ID}" type={TYPE} depart="0">\n' \
                         '        <route edges="{START} {END}"/>\n' \
                         '    </vehicle>'.format(ID = self.ego.agentid, TYPE = EGO_TYPE[0], START = start_edge, END = finish_edge)

        # populate environment with random vehicles departing at random times from random input edges
        npc_type_line = '<vType id={ID} accel={ACCEL} sigma="0.5" length={LENGTH}/>'.format(
            ID = NPC_TYPE[0], ACCEL = NPC_TYPE[1], LENGTH = NPC_TYPE[2], MAX = NPC_TYPE[3]
        )

        npc_routes = []
        self.npc_depart_times = random.sample(range(0, SPAWN_TIME_RANGE), random.randint(MIN_NPCS, MAX_NPCS))
        self.npc_depart_times.sort()

        for id in range(len(self.npc_depart_times)):
            npc_start_edge = random.choice([edge for edge in edges if "-" not in edge])
            npc_finish_edge = random.choice([edge for edge in edges if "-" in edge])
            npc_route_line = '<vehicle id="npc{ID}" type={TYPE} depart="{DEPART}">\n' \
                         '        <route edges="{START} {FINISH}"/>\n' \
                         '    </vehicle>'.format(ID=id, TYPE=NPC_TYPE[0], DEPART=self.npc_depart_times[id], START=npc_start_edge, FINISH=npc_finish_edge)
            npc_routes.append(npc_route_line)

        # write route file
        file = open(CONFIGS_PATH + "{route}".format(route=route), "w")

        file.write("<routes>\n")
        file.write("    " + ego_type_line + "\n\n")
        file.write("    " + npc_type_line + "\n\n")
        file.write("    " + ego_route_line + "\n\n")

        for route in npc_routes:
            file.write("    " + route + "\n\n")

        file.write("</routes>\n")
        file.close()

        # compile sumocfg
        # os.system('{configs}sumocompile.sh'.format(configs=CONFIGS_PATH))

        # start simulation
        libsumo.start(["{render}".format(render="sumo" if self.render_mode == "cli" else "sumo-gui"),
                       "-c",
                       "{configs}{config}sumocfg".format(configs=CONFIGS_PATH, config=network[:-7]),
                       "--lateral-resolution=3.0"])

        # trigger rerouting
        libsumo.vehicle_rerouteTraveltime(self.ego.agentid)
        for i in range(len(npc_routes)):
            if self.npc_depart_times[i] == 0:
                libsumo.vehicle_rerouteTraveltime("npc{num}".format(num=i))

        # advance to timestep 0

        libsumo.simulationStep()

        # build initial observation
        return self._get_obs()


    def step(self, action):
        # extract high level behaviour decision
        behaviour = action[0]

        # extract low level throttle decision
        throttle = action[1]

        # TODO: implement steering (irrelevant to SUMO)

        # reroute vehicles appearing at this time step
        for veh_id in self.npc_depart_times:
            libsumo.vehicle_rerouteTraveltime("npc{num}".format(num=veh_id))


    def render(self):
        pass

    def close(self):
        libsumo.simulation_close()