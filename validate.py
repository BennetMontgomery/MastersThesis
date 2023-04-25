from agents.proposed_agent import ProposedAgent
from os import listdir
import matplotlib.pyplot as plt

ego = ProposedAgent(
    agentid='ego',
    network=None,
    verbose=False
)

SAVE_TXT = True

validation_folder = './Validationconfgs/'
model_folder = './models/'

network_range = ["2023-04-20 19:19:53.228604_0", "2023-04-21 06:49:06.071330_final"]

# networks = [
#     "2023-03-27 19:15:40.016768_0",
#     "2023-03-27 19:29:54.540541_25",
#     "2023-03-27 19:45:40.170759_50",
#     "2023-03-27 20:01:27.384814_75",
#     "2023-03-27 20:20:28.372448_100",
#     "2023-03-27 20:40:07.452981_125",
#     "2023-03-27 20:56:23.658337_150",
#     "2023-03-27 21:17:46.927825_175",
#     "2023-03-27 21:35:24.424363_200",
#     "2023-03-27 21:56:23.288509_225",
#     "2023-03-27 22:17:50.791560_250",
#     "2023-03-27 22:39:25.877965_275",
#     "2023-03-27 23:02:21.462654_300",
#     "2023-03-27 23:17:20.402968_325",
#     "2023-03-27 23:36:56.338274_350",
#     "2023-03-27 23:53:00.688863_375",
#     "2023-03-28 00:08:58.583338_400",
#     "2023-03-28 00:30:18.069438_425",
#     "2023-03-28 00:53:47.791317_450",
#     "2023-03-28 01:15:37.241679_475",
#     "2023-03-28 01:38:12.170607_final"
# ]

# networks = [
#     "2023-04-18 20:23:32.801488_0",
#     "2023-04-18 20:39:27.805158_25",
#     "2023-04-18 20:58:31.102047_50",
#     "2023-04-18 21:17:44.433238_75",
#     "2023-04-18 21:34:12.973726_100",
#     "2023-04-18 21:49:09.690622_125",
#     "2023-04-18 22:08:18.821496_150",
#     "2023-04-18 22:24:40.247075_175",
#     "2023-04-18 22:43:03.343331_200",
#     "2023-04-18 22:59:12.089915_225",
#     "2023-04-18 23:14:27.994029_250",
#     "2023-04-18 23:34:27.733942_275",
#     "2023-04-18 23:50:34.077516_300",
#     "2023-04-19 00:07:32.625307_325",
#     "2023-04-19 00:21:52.632890_350",
#     "2023-04-19 00:39:02.626281_375",
#     "2023-04-19 00:58:54.680139_400",
#     "2023-04-19 01:15:39.449149_425",
#     "2023-04-19 01:34:31.026761_450",
#     "2023-04-19 01:54:23.048014_475",
#     "2023-04-19 02:12:42.158082_500",
#     # "2023-04-12 06:17:23.224193_525",
#     # "2023-04-12 06:35:01.782226_550",
#     # # "2023-04-12 06:49:31.130676_575",
#     # "2023-04-12 07:05:03.150739_600",
#     # # "2023-04-12 07:26:20.497576_625",
#     # "2023-04-12 07:44:04.874659_650",
#     # # "2023-04-12 08:04:20.098612_675",
#     # "2023-04-12 08:22:16.460053_700",
#     # # "2023-04-12 08:45:23.378641_725",
#     # "2023-04-12 09:05:20.368692_final"
# ]

networks = listdir(model_folder)
networks.sort()
networks = networks[networks.index(network_range[0]):networks.index(network_range[1]) + 1]

# domain = [i*100 for i in range(0, 21)]
domain = [i*100 for i in range(len(networks))]
configs=[
    "magic.net.xml",
    "simple.net.xml",
    "threelane.net.xml",
    "twolane.net.xml",
    "unrealistic.net.xml"
]
npcs=[
    # [i for i in range(0, 23)],
    [2, 5, 9, 11, 12, 15],
    [2, 5, 9, 11, 12, 15],
    [i for i in range(0, 24)],
    # [2, 5, 9, 11, 12, 15],
    [1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 21, 22, 23],
    [i for i in range(0, 24)],
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

# print(graphically_validate(3, "2022-11-16 02:50:48.228403_1900"))
plot_validate()