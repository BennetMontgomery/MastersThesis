import tensorflow as tf


class Encoder(tf.keras.layers.Layer):
    def __init__(self, attention_layer_width, num_heads, feed_forward_params, rate, agent_len, in_range=200):
        super(Encoder, self).__init__()

        self.attention_layer_width = attention_layer_width
        self.embedder = tf.keras.layers.Embedding(in_range, attention_layer_width)
        self.agent_len = agent_len

        self.multi_head = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=attention_layer_width,
            dropout=rate
        )
        self.feed_forward = tf.keras.Sequential([
            tf.keras.layers.Dense(feed_forward_params, activation='relu'),
            tf.keras.layers.Dense(attention_layer_width)
        ])

        self.multi_head_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.feed_forward_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout_attention = tf.keras.layers.Dropout(rate)
        self.dropout_feedforward = tf.keras.layers.Dropout(rate)

    def call(self, agents, training=True, mask=None):
        npc_embedding = tf.reshape(self.embedder(agents[1]), [len(agents[1]) * self.agent_len, self.attention_layer_width])
        ego_embedding = self.embedder(agents[0])
        # project ego_embedding to match npc_embedding batchsize (required by keras)
        ego_embedding = tf.repeat(tf.expand_dims(tf.reduce_sum(ego_embedding, 0), axis=0), [len(npc_embedding)], axis=0)

        # batchify input
        npc_embedding = tf.expand_dims(npc_embedding, axis=0)
        ego_embedding = tf.expand_dims(ego_embedding, axis=0)


        attention = self.multi_head(
            query=ego_embedding,
            key=npc_embedding,
            value=npc_embedding,
            training=training,
            attention_mask=mask
        )
        attention = self.dropout_attention(attention, training=training)
        a_output = self.multi_head_norm(npc_embedding + attention)

        f_output = self.feed_forward(a_output)
        f_output = self.dropout_feedforward(f_output, training=training)
        output = self.feed_forward_norm(a_output + f_output)

        return output


class AttentionPooler(tf.keras.layers.Layer):
    def __init__(self, attention_layer_width, layer_params):
        super(AttentionPooler, self).__init__()

        self.attention_layer_width = attention_layer_width

        self.pooler_input = tf.keras.layers.Dense(attention_layer_width, activation='relu')
        self.pooling_layers = [tf.keras.layers.Dense(layer, activation='relu') for layer in layer_params]

    def call(self, inputs):
        print(inputs)
        inputs = tf.squeeze(inputs)
        outputs = []

        for tensor in inputs:
            batch = tf.expand_dims(tensor, axis=0)

            output = self.pooler_input(batch)

            for layer in self.pooling_layers:
                output = layer(output)

            outputs.append(output)

        pooled = tf.reduce_sum(tf.convert_to_tensor(outputs), axis=0)

        return pooled
