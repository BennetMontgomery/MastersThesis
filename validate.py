from agents.proposed_agent import ProposedAgent
import matplotlib.pyplot as plt

ego = ProposedAgent(
    agentid='ego',
    network=None,
    verbose=False
)

validation_folder = './Validationconfgs/'
domain = [i*100 for i in range(0, 21)]
networks = [
    "2022-11-14 17:16:56.830649_0",
    "2022-11-14 19:05:20.309803_100",
    "2022-11-14 20:53:07.102073_200",
    "2022-11-14 22:37:21.469217_300",
    "2022-11-15 00:22:31.294851_400",
    "2022-11-15 02:21:50.231955_500",
    "2022-11-15 04:05:55.387547_600",
    "2022-11-15 05:52:52.224134_700",
    "2022-11-15 07:36:02.542764_800",
    "2022-11-15 09:23:02.249759_900",
    "2022-11-15 11:11:09.706469_1000",
    "2022-11-15 12:58:22.014795_1100",
    "2022-11-15 14:45:21.589189_1200",
    "2022-11-15 16:33:06.997286_1300",
    "2022-11-15 18:09:34.794048_1400",
    "2022-11-15 19:47:09.304288_1500",
    "2022-11-15 21:41:01.146437_1600",
    "2022-11-15 23:20:19.548506_1700",
    "2022-11-16 01:04:06.300645_1800",
    "2022-11-16 02:50:48.228403_1900",
    "2022-11-16 04:46:51.278430_final"
]
configs=[
    "magic.net.xml",
    "simple.net.xml",
    "threelane.net.xml",
    "twolane.net.xml",
    "unrealistic.net.xml"
]
npcs=[
    [1, 2, 4, 15],
    [2, 5, 9, 11, 12, 15],
    [0, 1, 2, 4, 6, 8, 10, 12, 13, 15],
    [1, 2, 4, 5, 10, 12, 14],
    [1, 5, 6, 9, 10, 11, 13, 15]
]

def plot_validate():
    for i in range(len(configs)):
        rewards = []
        rewards_matrix = {
            "time":[],
            "ilc":[],
            "motion":[],
            "reversing":[],
            "collision":[],
            "goal":[],
            "exit":[],
            "timeout":[]
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

        for key in rewards_matrix.keys():
            plt.plot(domain, rewards_matrix[key])
            plt.title(f"reward for {key} in {configs[i]} over time")
            plt.savefig(f"./{key}_{configs[i]}.png")
            plt.close()

def graphically_validate(confg_idx=0, network="2022-11-14 19:05:20.309803_100"):
    reward, reward_matrix = ego.validate(configs[confg_idx], validation_folder, npcs=npcs[confg_idx], network=network, graphical_mode=True, split_reward=True)

    return reward, reward_matrix

# print(graphically_validate(3, "2022-11-16 02:50:48.228403_1900"))
plot_validate()