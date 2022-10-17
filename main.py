from agents.proposed_agent import ProposedAgent

ego = ProposedAgent(
    agentid='ego',
    network=None,
    verbose=True
)

ego.train_nets(episodes=2000)

