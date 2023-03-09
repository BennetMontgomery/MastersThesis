from agents.proposed_agent import ProposedAgent
import matplotlib.pyplot as plt

ego = ProposedAgent(
    agentid='ego',
    network=None,
    verbose=False
)

validation_folder = './Validationconfgs/'

networks = [
    "2023-03-07 08:13:55.564151_final"
]
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
    [1, 2, 4, 15],
    [2, 5, 9, 11, 12, 15],
    [0, 1, 2, 4, 6, 8, 10, 12, 13, 15],
    [1, 2, 4, 5, 10, 12, 14],
    [1, 5, 6, 9, 10, 11, 13, 15],
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