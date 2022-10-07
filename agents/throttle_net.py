import tensorflow as tf
import behaviour_net

class ThrottleNet(behaviour_net.BehaviourNet):
    def __init__(self, static_input_size, variable_input_size,
                 attention_in_d, q_layers, pooler_layers, attention_out_ff, attention_heads,
                 memory_cap, e_decay=0.001, dropout_rate=0.01):
        super(ThrottleNet, self).__init__(static_input_size, variable_input_size, attention_in_d, q_layers,
                                          pooler_layers, attention_out_ff, attention_heads, memory_cap, e_decay,
                                          dropout_rate)

    def call(self, inputs: list[list[int]], training=True, attention_mask=None):
        pass