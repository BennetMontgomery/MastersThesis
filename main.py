from agents.proposed_agent import ProposedAgent
import wandb

# ego = ProposedAgent(
#     agentid='ego',
#     network=None,
#     verbose=False
# )

# wandb.init(project="wandb-test")

# ego.train_nets(episodes=2000)

hyperparameter_scans = {
    "alpha" : [0.001, 0.0001],
    "gamma" : [0.9, 0.95],
    "b_updates" : [10, 25, 50],
    "t_updates" : [10, 25, 50]
}

for alpha in hyperparameter_scans["alpha"]:
    for gamma in hyperparameter_scans["gamma"]:
        for b_updates in hyperparameter_scans["b_updates"]:
            for t_updates in hyperparameter_scans["t_updates"]:
                ego = ProposedAgent(agentid="ego", network=None, verbose=False)
                ego.alpha = alpha
                ego.gamma = gamma
                ego.update_freq_b = b_updates
                ego.update_freq_t = t_updates
                
                wandb.init(project="param-scan", id=f"alpha{alpha}-gamma{gamma}-update_b{b_updates}-update_t{t_updates}")
                
                ego.train_nets(episodes=500)
                
                wandb.finish()