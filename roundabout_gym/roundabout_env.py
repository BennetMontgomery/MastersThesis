import gym
from gym import spaces
from agents.agent import Agent
from parameters.simulator_params import *
import libsumo
import os
import random

# Constants
CONFIGS_PATH="./Sumoconfgs/"
EGO_TYPE = ['"agenttype"', '"4.0"', '"7.0"', '"5"', '"30"', '"red"']
NPC_TYPE = ['"npctype"', '"0.8"', '"5"', '"14"']
MIN_NPCS = minimum_npcs
MAX_NPCS = maximum_npcs
COLLISION_REWARD = collision_reward
GOAL_REWARD = 100
TIMEOUT = 100
# TIMEOUT_REWARD = -10
TIMESTEP_REWARD = -5
ILLEGAL_LANE_CHANGE_REWARD = illegal_lane_change_reward
SPAWN_TIME_RANGE = spawn_time_range
EXIT_REWARD = exit_reward
EGO_IN_OBS = True


class RoundaboutEnv(gym.Env):
    metadata = {"render_modes" : ["sumo-gui", "cli"]}

    def __init__(self, ego: Agent, render_mode: str = "cli"):
        assert render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.ego = ego # object reference to the ego agent

        self.ego_last_edge = None
        self.ego_goal = None
        self.prev_obs = None

        # Action space
        #   Index 0: Behaviour at time step
        #   Index 2: Acceleration x0.5 at time step + 14
        self.action_space = spaces.MultiDiscrete([3, 22])
        # lane change variables
        self.changing_lanes = False
        self.start_lane = None
        self.target_lane = None

        self.prev_accel_val = 0

        self.npc_depart_times = []

        self.network = None
        self.networks = []



    ''' private method for collating observations for processing '''
    def _get_obs(self):
        # STATIC OBSERVATION VECTOR
        # number: visible vehicles
        # speed: current vehicle speed
        # accel: current vehicle acceleration
        # laneid: current lane the vehicle is in
        # lanepos: current longitudinal lane position
        # dti: distance to next intersection
        # lanes: adjacent lane data

        lanes_available = [0, 0]

        if libsumo.edge_getLaneNumber(libsumo.vehicle_getRoadID(self.ego.agentid)) > libsumo.vehicle_getLaneIndex(self.ego.agentid):
            lanes_available[0] = 1

        if libsumo.vehicle_getLaneIndex(self.ego.agentid) > 0:
            lanes_available[1] = 1

        self.ego_last_edge = libsumo.vehicle_getRoadID(self.ego.agentid)

        obs = {
            "number": len(self.ego.view),
            "speed": libsumo.vehicle_getSpeed(self.ego.agentid),
            "accel": libsumo.vehicle_getAcceleration(self.ego.agentid),
            "laneid": libsumo.vehicle_getLaneIndex(self.ego.agentid),
            "lanepos": libsumo.vehicle_getLanePosition(self.ego.agentid),
            "dti": libsumo.lane_getLength(libsumo.vehicle_getLaneID(self.ego.agentid))
                   - libsumo.vehicle_getLanePosition(self.ego.agentid),
            "lanes": lanes_available
        }

        # append vehicle tokens
        # speed: speed of vehicle
        # lane id: lane index of vehicle
        # lanepos: longitudinal lane position of vehicle in lane index
        # inedge: 1 if on same edge else 0
        # intarget: 1 if on target link, else 0
        # insource: 1 if on a lane linking to the current target, else 0
        vehicle_list = [
            {'ego': {
                "speed": libsumo.vehicle_getSpeed(self.ego.agentid),
                "laneid": libsumo.vehicle_getLaneIndex(self.ego.agentid),
                "lanepos": libsumo.vehicle_getLanePosition(self.ego.agentid),
                "inedge": 1,
                "intarget": 0,
                "insource": 1
            }}
        ] if EGO_IN_OBS else []
        for vehicle in self.ego.view:
            target_edges = [i[0] for i in libsumo.lane_getLinks(libsumo.vehicle_getLaneID(vehicle))]

            for i in range(len(target_edges)):
                target_edges[i] = target_edges[i].split("_")[0]

            vehicle_list.append({vehicle + str(self.ego.view.index(vehicle)): {
                "speed": libsumo.vehicle_getSpeed(vehicle),
                "laneid": libsumo.vehicle_getLaneIndex(vehicle),
                "lanepos": libsumo.vehicle_getLanePosition(vehicle),
                "inedge": 1 if libsumo.vehicle_getRoadID(vehicle) == libsumo.vehicle_getRoadID(self.ego.agentid) else 0,
                "intarget": 1 if libsumo.vehicle_getRoadID(vehicle) ==
                                libsumo.vehicle_getRoute(self.ego.agentid)[libsumo.vehicle_getRouteIndex(self.ego.agentid)+1] else 0,
                "insource": 1 if libsumo.vehicle_getRoute(self.ego.agentid)[libsumo.vehicle_getRouteIndex(self.ego.agentid)+1]
                                in target_edges else 0
            }})

        if len(vehicle_list) > 0:
            obs = obs | {"vehicles": vehicle_list}

        # append light data
        light_list = []
        for light in libsumo.trafficlight_getIDList():
            if libsumo.vehicle_getLaneID(self.ego.agentid) is libsumo.trafficlight_getControlledLanes(light):
                if libsumo.trafficlight_getRedYellowGreenState(light) in ['r', 'u']:
                    light_val = 0
                elif libsumo.trafficlight_getRedYellowGreenState(light) in ['y', 'Y']:
                    light_val = 1
                elif libsumo.trafficlight_getRedYellowGreenState(light) in ['g', 'G']:
                    light_val = 2
                else:
                    light_val = None # traffic light is off and is functionally a stop sign

                if light_val is not None:
                    light_list.append({light: light_val})

        if len(light_list) > 0:
            obs = obs | {"lights": light_list}

        return obs

    def reset(self, seed=None, return_info=False, options=None):
        # Seed random generator
        random.seed()
        super().reset(seed=seed)

        ''' Open configuration files and select roundabout model '''
        sumoconfigs = os.listdir("{configs}".format(configs=CONFIGS_PATH))
        if len(self.networks) == 0:
            self.networks = [entry for entry in sumoconfigs if entry.endswith(".net.xml")]

        # select random network
        if options is None:
            self.network = random.choice(self.networks)
            self.networks.remove(self.network)
            route = "{net}rou.xml".format(net=self.network[:-7])
            num_npcs_0 = 0
        else:
            # select validation network
            self.network = options[0]
            route = "{net}rou.xml".format(net=self.network[:-7])
            self.npc_depart_times = options[2]
            num_npcs_0 = 0
            for npc in self.npc_depart_times:
                if npc == 0:
                    num_npcs_0 += 1

        print(self.network)

        if options is None:
            # place ego agent at lane 0 in a random edge at timestep 0
            file = open(CONFIGS_PATH + "{net}".format(net=self.network))
            edges = []
            for line in file:
                line = line.strip()
                if line is not None and "edge id=" in line and "function=\"internal\"" not in line:
                    edges.append(line.split("edge id=")[1].split("\"")[1].split("\"")[0])

            file.close()

        # always select a starting edge pointing towards a roundabout
            start_edge = random.choice([edge for edge in edges if "-" not in edge and "R" not in edge])
            self.ego_last_edge = start_edge

            # select a finishing edge pointing away from the roundabout
            finish_edge = random.choice([edge for edge in edges if "-" in edge])
            self.ego_goal = finish_edge

            ego_type_line = '<vType id={ID} accel={ACCEL} emergencyDecel={EDECEL} sigma="0" length={LENGTH} maxspeed={MAX} color={COLOUR}/>'.format(
                ID = EGO_TYPE[0], ACCEL = EGO_TYPE[1], EDECEL = EGO_TYPE[2], LENGTH = EGO_TYPE[3], MAX = EGO_TYPE[4], COLOUR = EGO_TYPE[5]
            )
            ego_route_line = '<vehicle id="{ID}" type={TYPE} depart="0">\n' \
                             '        <route edges="{START} {END}"/>\n' \
                             '        <param key="has.ssm.device" value="true"/>\n' \
                             '        <param key="device.ssm.measures" value="DRAC"/>\n' \
                             '    </vehicle>'.format(ID = self.ego.agentid, TYPE = EGO_TYPE[0], START = start_edge, END = finish_edge)

            # populate environment with random vehicles departing at random times from random input edges
            npc_type_line = '<vType id={ID} accel={ACCEL} sigma="0.5" length={LENGTH}/>'.format(
                ID = NPC_TYPE[0], ACCEL = NPC_TYPE[1], LENGTH = NPC_TYPE[2], MAX = NPC_TYPE[3]
            )

            npc_routes = []
            self.npc_depart_times = random.sample(range(0, SPAWN_TIME_RANGE), random.randint(MIN_NPCS, MAX_NPCS))
            self.npc_depart_times.sort()
            for npc in self.npc_depart_times:
                if npc == 0:
                    num_npcs_0+=1

            for id in range(len(self.npc_depart_times)):
                npc_start_edge = random.choice([edge for edge in edges if "-" not in edge])
                npc_finish_edge = random.choice([edge for edge in edges if "-" in edge])
                npc_route_line = '<vehicle id="npc{ID}" type={TYPE} depart="{DEPART}">\n' \
                             '        <route edges="{START} {FINISH}"/>\n' \
                             '        <param key="has.ssm.device" value="true"/>\n' \
                             '        <param key="device.ssm.measures" value="DRAC"/>\n' \
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

        # start simulation
        if options is None:
            libsumo.start(["{render}".format(render="sumo" if self.render_mode == "cli" else "sumo-gui"),
                           "-c",
                           "{configs}{config}sumocfg".format(configs=CONFIGS_PATH, config=self.network[:-7]),
                           "--lateral-resolution=3.0"])
        else:
            libsumo.start(["{render}".format(render="sumo" if self.render_mode == "cli" else "sumo-gui"),
                           "-c",
                           "{configs}{config}sumocfg".format(configs=options[1], config=self.network[:-7]),
                           "--lateral-resolution=3.0"])

        # trigger rerouting
        libsumo.vehicle_rerouteTraveltime(self.ego.agentid)
        for i in range(num_npcs_0):
            libsumo.vehicle_rerouteTraveltime("npc{num}".format(num=i))

        # advance to timestep 0

        libsumo.simulationStep()

        # build initial observation
        if options is not None:
            self.ego_goal = libsumo.vehicle_getRoute(self.ego.agentid)[-1]

        self.prev_obs = self._get_obs()
        self.prev_accel_val = 0
        self.last_distance = 0
        return self.prev_obs


    def step(self, action, split_reward=False):
        # set timestep reward to 0
        if split_reward:
            reward_matrix={
                "time": 0,
                "ilc": 0,
                "motion": 0,
                "goal": 0,
                # "drac":0,
                "speed":0,
            }

        reward = 0

        # extract high level behaviour decision
        behaviour = action[0]

        # extract low level throttle decision
        throttle = action[1]

        # print(f"Behaviour: {behaviour} Throttle: {throttle}")

        # reroute vehicles appearing at this time step
        for i in range(len(self.npc_depart_times)):
            if self.npc_depart_times[i] == libsumo.simulation_getTime():
                libsumo.vehicle_rerouteTraveltime("npc{num}".format(num=i))

        #### Apply action
        # apply behaviour decision
        if behaviour == 0: # change lane left
            # sanity check: only change lanes if a lane exists
            if libsumo.edge_getLaneNumber(libsumo.vehicle_getRoadID(self.ego.agentid)) - 1 \
                    > libsumo.vehicle_getLaneIndex(self.ego.agentid):
                # continue in progress lane change
                if self.changing_lanes:
                    # check if vehicle has reached target lane
                    if libsumo.vehicle_getLaneIndex(self.ego.agentid) == self.target_lane:
                        # lanewise centre, and terminate if successful
                        self.changing_lanes = not self.ego.lanewise_centre()
                        if not self.changing_lanes:
                            self.source_lane = None
                            self.target_lane = None
                # if no lange change in progress, initiate lane change
                else:
                    self.changing_lanes = True
                    self.source_lane = libsumo.vehicle_getLaneIndex(self.ego.agentid)
                    self.target_lane = libsumo.vehicle_getLaneIndex(self.ego.agentid) + 1
                    self.ego.change_lane(self.target_lane)
                
                reward -= 5
                if split_reward:
                    reward_matrix["ilc"] -= 5
        elif behaviour == 1: # change lane right
            # sanity check: only change lanes if a lane exists
            if libsumo.vehicle_getLaneIndex(self.ego.agentid) > 0:
                # continue in progress lane change
                if self.changing_lanes:
                    # check if vehicle has reached target lane
                    if libsumo.vehicle_getLaneIndex(self.ego.agentid) == self.target_lane:
                        # lanewise centre, and terminate if successful
                        self.changing_lanes = not self.ego.lanewise_centre()
                        if not self.changing_lanes:
                            self.source_lane = None
                            self.target_lane = None
                # if no lange change in progress, initiate lane change
                else:
                    self.changing_lanes = True
                    self.source_lane = libsumo.vehicle_getLaneIndex(self.ego.agentid)
                    self.target_lane = libsumo.vehicle_getLaneIndex(self.ego.agentid) - 1
                    self.ego.change_lane(self.target_lane)
                
                reward -= 5
                if split_reward:
                    reward_matrix["ilc"] -= 5
#         elif behaviour == 2: # follow leader
#             # give small reward for not changing lanes
#             #reward += 0.5
#             # reward += 5
            
#             if split_reward:
#                 # reward_matrix["ilc"] += 0.5
#                 reward_matrix["ilc"] += 5
        elif behaviour != 2:
            raise ValueError("Incorrect first index action value")

        # convert throttle action value to speed change value
        accel_val = (throttle - 14)/2
        new_speed = libsumo.vehicle_getSpeed(self.ego.agentid) + accel_val

        if abs(accel_val - self.prev_accel_val) > 3:
            # reward -= abs(accel_val - self.prev_accel_val) # unsmooth motion penalty
            reward -= 8

            if split_reward:
                reward_matrix["motion"] -= 8
        elif abs(accel_val - self.prev_accel_val) > 2:
            reward -= 4
            
            if split_reward:
                reward_matrix["motion"] -= 4
        elif abs(accel_val - self.prev_accel_val) > 1:
            reward -= 2
            
            if split_reward:
                reward_matrix["motion"] -= 2
        
        # relog accel
        self.prev_accel_val = accel_val
        
        # apply throttle decision
        self.ego.change_speed(new_speed, accel_val)
        
        # apply speed reward: driving within the speed limit of 50 km/h
#         if abs(new_speed - 14) < 2:
#             reward += 0.5
            
#             if split_reward:
#                 reward_matrix["speed"] += 0.5
        # try:
        #     speed_reward = 15*max(0, min(new_speed/13, 13/new_speed))
        # except ZeroDivisionError:
        #     speed_reward = 0
        
        if new_speed > 0 or new_speed < -13:
            speed_reward = 15*(min(new_speed/13, 13/new_speed)-1)
        else:
            speed_reward = 15*((new_speed/13) - 1)
            
        reward += speed_reward
        
        if split_reward:
            reward_matrix["speed"] += speed_reward

        # increment simulation
        libsumo.simulationStep()

        # check if now in terminal state
        if self.ego.agentid in libsumo.simulation_getCollidingVehiclesIDList():
            # terminate with 0 goal and time rewards
            if split_reward:
                reward_matrix["goal"] += -100
                reward_matrix["time"] = -100
                
                return self.prev_obs, reward, reward_matrix, True, None

            return self.prev_obs, reward, True, None
        elif self.ego.agentid not in libsumo.vehicle_getIDList() and self.ego_goal == self.ego_last_edge:
            # reward for succesfully exiting goal state
            reward += GOAL_REWARD
            # reward for time taken
            # expected_time = self.last_distance/13
            
#             # grant maximum time reward if minimum expected time taken
#             if libsumo.simulation_getTime() <= expected_time:
#                 time_reward = 100*GOAL_REWARD
#             else:
#                 time_reward = 100*(expected_time*(10/11))/(libsumo.simulation_getTime() - (expected_time*(10/11)))

#             reward += time_reward
                                                       
            if split_reward:
                reward_matrix["goal"] += GOAL_REWARD
                # reward_matrix["time"] += time_reward           

                return self.prev_obs, reward, reward_matrix, True, None

            return self.prev_obs, reward, True, None
        elif self.ego.agentid not in libsumo.vehicle_getIDList():
            # reward for exiting system without reaching goal
            if split_reward:
                return self.prev_obs, reward, reward_matrix, True, None

            return self.prev_obs, reward, True, None
        elif libsumo.simulation_getTime() > TIMEOUT:
            # reward for getting stuck
            reward += -300

            if split_reward:
                reward_matrix["goal"] = -300

                return self.prev_obs, reward, reward_matrix, True, None

            return self.prev_obs, reward, True, None\

        # apply drac reward
#         for vehicle in libsumo.vehicle_getIDList():
#             if libsumo.vehicle_getParameter(vehicle, "device.ssm.maxDRAC") == '':
#                 reward -= 1

#                 if split_reward:
#                     reward_matrix["drac"] -= 1

        # apply timestep penalty
        reward += TIMESTEP_REWARD
        
        if split_reward:
            reward_matrix["time"] += TIMESTEP_REWARD

        self.prev_obs = self._get_obs()
        self.last_distance = libsumo.vehicle_getDistance(self.ego.agentid)
        self.ego_last_edge = libsumo.vehicle_getRoadID(self.ego.agentid)

        if split_reward:
            # reward_matrix["time"] += TIMESTEP_REWARD

            return self.prev_obs, reward, reward_matrix, False, None

        return self.prev_obs, reward, False, None

    def sample(self):
        return self._get_obs()

    def render(self):
        pass

    def close(self):
        libsumo.simulation_close()