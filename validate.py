from agents.proposed_agent import ProposedAgent
from os import listdir
import matplotlib.pyplot as plt
import libsumo

ego = ProposedAgent(
    agentid='ego',
    network=None,
    verbose=False
)

SAVE_TXT = True

validation_folder = './Validationconfgs/'
model_folder = './models/'

network_range = ["2023-05-21 17:45:03.610686_0", "2023-05-22 06:08:18.506414_final"]

networks = listdir(model_folder)
networks.sort()
networks = networks[networks.index(network_range[0]):networks.index(network_range[1]) + 1]

domain = [i*10 for i in range(len(networks))]

configs=[
    "magic.net.xml",
    "simple.net.xml",
    "threelane.net.xml",
    "twolane.net.xml",
    "unrealistic.net.xml"
]
npcs=[
    [i for i in range(0, 23)],
    [0, 1, 2, 3, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 17, 18, 19, 20, 21, 22, 23],
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 20, 22, 23],
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 18, 19, 20, 21, 22, 23],
    [0, 1, 2, 3, 5, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22],
]

def plot_validate():
    for i in range(len(configs)):
        rewards = []
        rewards_matrix = {
            "time":[],
            "ilc":[],
            "motion":[],
            "goal":[],
            "drac":[],
            "speed":[],
        }

        for network in networks:
            reward, reward_matrix = ego.validate(configs[i], validation_folder, npcs=npcs[i], network=network, split_reward=True)

            rewards.append(reward)

            for key in reward_matrix.keys():
                rewards_matrix[key].append(reward_matrix[key])

        plt.plot(domain, rewards)
        plt.title(f"total {configs[i]} performance over time")
        plt.savefig(f"./{configs[i]}.png")
        plt.close()
        
        if SAVE_TXT:
            print(list(map(str, rewards)))
            file = open(f"{configs[i]}_performance.txt", "w")
            file.writelines(list(map(lambda a : str(a) + '\n', rewards)))
            file.close()
            

        for key in rewards_matrix.keys():
            plt.plot(domain, rewards_matrix[key])
            plt.title(f"reward for {key} in {configs[i]} over time")
            plt.savefig(f"./{key}_{configs[i]}.png")
            plt.close()
            
            if SAVE_TXT:
                file = open(f"{configs[i]}_reward_for_{key}_over time.txt", "w")
                file.writelines(list(map(lambda a : str(a) + '\n', rewards_matrix[key])))
                file.close()
            

def graphically_validate(confg_idx=0, network="2022-11-14 19:05:20.309803_100"):
    reward, reward_matrix = ego.validate(configs[confg_idx], validation_folder, npcs=npcs[confg_idx], network=network, graphical_mode=True, split_reward=True)

    return reward, reward_matrix


def sumo_score(sigma):
    rewards = []
    rewards_matrix = {
        "time": [],
        "ilc": [],
        "motion": [],
        "goal": [],
        # "drac": [],
        "speed": [],
    }

    for i in range(len(configs)):
        # initiate simulation
        libsumo.start(["sumo", "-c", f"{validation_folder}{configs[i][:-7]}sumocfg", "--lateral-resolution=3.0"])

        # reroute vehicles
        libsumo.vehicle_rerouteTraveltime("ego")
        libsumo.vehicletype_setImperfection("agenttype", sigma)
        for npc in range(len(npcs[i])):
            if npcs[i][npc] == 0:
                libsumo.vehicle_rerouteTraveltime(f"npc{npc}")

        # step through simulation
        libsumo.simulationStep()

        # initial speed
        old_speed = libsumo.vehicle_getSpeed("ego")

        # initial lane
        old_lane = libsumo.vehicle_getLaneIndex("ego")

        terminated = False
        step = 0
        reward = 0
        reward_matrix = {
            "time": 0,
            "ilc": 0,
            "motion": 0,
            "goal": 0,
            # "drac": 0,
            "speed": 0,
        }
        while not terminated:

            # reroute vehicles appearing in this time step
            if step+1 in npcs[i]:
                testindex = npcs[i].index(step+1)
                libsumo.vehicle_rerouteTraveltime(f"npc{testindex}")

            # deduce action
            action = [0, 0]
            new_lane = libsumo.vehicle_getLaneIndex("ego")
            if new_lane > old_lane:
                action[0] = 0
            elif new_lane > old_lane:
                action[0] = 1
            else:
                action[0] = 2

            new_speed = libsumo.vehicle_getSpeed("ego")
            action[1] = ((new_speed - old_speed) - 14)/2

            # calculate reward
            # time
            reward -= 5
            reward_matrix["time"] -= 5

            # lane change
            reward -= 5 if action[0] != 2 else 0
            reward_matrix["ilc"] -= 5 if action[0] != 2 else 0

            # motion
            reward -= 8 if abs(new_speed - old_speed) > 3 else 0

            # speed
            if new_speed > 0 or new_speed < -13:
                reward += 15 * (min(new_speed / 13, 13 / new_speed) - 1)
                reward_matrix["speed"] += 15 * (min(new_speed / 13, 13 / new_speed) - 1)
            else:
                reward += 15 * ((new_speed / 13) - 1)
                reward_matrix["speed"] += 15 * ((new_speed / 13) - 1)

            libsumo.simulationStep()

            # goal
            # - timeout
            if libsumo.simulation_getTime() > 100:
                reward -= 300
                reward_matrix["goal"] -= 300
                terminated = True
                break
            # - collision
            elif "ego" in libsumo.simulation_getCollidingVehiclesIDList():
                reward -= 200
                reward_matrix["goal"] -= 200
                terminated = True
                break
            # - reaching goal
            elif "ego" not in libsumo.vehicle_getIDList():
                reward += 100
                reward_matrix["goal"] += 100
                terminated = True
                break

            old_speed = new_speed
            old_lane = new_lane
            step += 1

        rewards.append(reward)
        for key in reward_matrix.keys():
            rewards_matrix[key].append(reward_matrix[key])

        libsumo.close()

    return rewards, rewards_matrix

# print(graphically_validate(3, "2022-11-16 02:50:48.228403_1900"))
# plot_validate()
test_scores = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
for score in test_scores:
    print(f"Scores for imperfection {score}")
    rewards, rewards_matrix = sumo_score(score)
    print("AGGREGATE")
    print(rewards)
    print("MEAN")
    print(sum(rewards)/len(rewards))
    print("SPLIT")
    print(rewards_matrix)
    print('')