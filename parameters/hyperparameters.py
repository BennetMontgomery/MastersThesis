# neural network hyperparameters file
# REINFORCEMENT LEARNING
gamma = 0.9
b_action_space_size = 3
t_action_space_size = 22
variable_input_size = 6
static_input_size = 8

# DEEP-LEARNING
# optimizer
alpha = 0.001

# HDQN
b_q_layers = [1024, 512, b_action_space_size]
t_q_layers = [1024, 512, t_action_space_size]
b_duration = 3
t_replay_size = 8000
b_replay_size = 8000
e_decay = 10e-3

# Transformers
update_freq_b = 50
num_heads_b = 32
update_freq_t = 50
num_heads_t = 32
b_feed_forward = 512
t_feed_forward = 512
b_d_model = 1024
t_d_model = 1024