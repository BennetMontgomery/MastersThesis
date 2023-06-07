from agents.proposed_agent import ProposedAgent
import wandb

ego = ProposedAgent(
    agentid='ego',
    network=None,
    verbose=False
)

wandb.init(project="wandb-test")

ego.train_nets(episodes=2000)

