from agents.proposed_agent import ProposedAgent

ego = ProposedAgent(
    agentid='ego',
    network=None,
    verbose=False
)

ego.train_nets(episodes=2000)

