import tensorflow as tf

class Encoder(tf.keras.layers.Layer):
    def __init__(self, attention_layer_width, num_heads, feed_forward_params, rate, agent_len, in_range=200):
        super(Encoder, self).__init__()

        self.attention_layer_width = attention_layer_width
        self.embedder = tf.keras.layers.Embedding(in_range, attention_layer_width, mask_zero=True)
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
        # ego_embedding = self.embedder(tf.convert_to_tensor(agents[0]))
        # npc_embedding = self.embedder(tf.convert_to_tensor(agents[1:]))
        ego_embedding = self.embedder(tf.expand_dims(tf.convert_to_tensor([obs[0] for obs in agents]),axis=1))
        npc_embedding = self.embedder(tf.convert_to_tensor([obs[1:] for obs in agents]))


        # batchify
        # if tf.rank(npc_embedding) < 4:
        #     ego_embedding = tf.expand_dims(tf.expand_dims(ego_embedding, axis=0), axis=0)
        #     npc_embedding = tf.expand_dims(npc_embedding, axis=0)
        # else:
        #     # batched by batch sampling already
        #     ego_embedding = tf.repeat(tf.expand_dims(ego_embedding, axis=0), tf.shape(npc_embedding)[0], axis=0)

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
    def __init__(self, attention_layer_width, layer_param, static_input_size):
        super(AttentionPooler, self).__init__()

        self.attention_layer_width = attention_layer_width

        self.pooler = tf.keras.Sequential([
            tf.keras.layers.Dense(attention_layer_width, input_shape=(static_input_size,attention_layer_width), activation='relu'),
            tf.keras.layers.Dense(layer_param)
        ])

        # self.pooler_input = tf.keras.layers.Dense(attention_layer_width, input_shape=input_s, activation='relu')
        # self.pooling_layers = [tf.keras.layers.Dense(layer, activation='relu') for layer in layer_params]

    def call(self, inputs):
        # inputs = tf.squeeze(inputs)
        outputs = []

        for batch in inputs:
            output_batch = []
            for tensor in batch:
                output = self.pooler(tf.expand_dims(tensor, axis=0))
                output_batch.append(output)

            pooled = tf.squeeze(tf.reduce_sum(output_batch, axis=0))
            outputs.append(pooled)
            # # if tf.rank(tensor) < 3:
            # #     tensor = tf.expand_dims(tensor, axis=0)
            #
            # output = self.pooler(tensor)
            #
            # # output = self.pooler_input(batch)
            # #
            # # for layer in self.pooling_layers:
            # #     output = layer(output)
            #
            # outputs.append(output)

        # pooled = tf.reduce_sum(tf.reduce_sum(tf.reduce_sum(tf.convert_to_tensor(outputs), axis=1), axis=0), axis=0)
        # pooled = tf.expand_dims(tf.reduce_sum(tf.reduce_sum(tf.convert_to_tensor(outputs), axis=1), axis=0), axis=0)

        return tf.convert_to_tensor(outputs,dtype='float32')
