import tensorflow as tf


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, layer_width, heads):
        super(MultiHeadAttention, self).__init__()
        self.heads = heads
        self.layer_width = layer_width

        self.qlayer = tf.keras.layers.Dense(layer_width)
        self.klayer = tf.keras.layers.Dense(layer_width)
        self.vlayer = tf.keras.layers.Dense(layer_width)

        self.dense = tf.keras.layers.Dense(layer_width)

    def scaled_dot_product(self, q, k, v, mask=None):
        '''
        Attention function used by MHA. Structure:
        Q -\
        K --> MatMul -> Scale -> Mask -> SoftMax -\
        V ---------------------------------------------> Matmul -> Attention vector

        :param q: query
        :param k: key
        :param v: value
        :param mask: mask vector
        :return: attention vector
        '''

        # Q-\
        # K --> MatMul
        matmul_qk = tf.matmul(q, k, transpose_b=True)

        # MatMul -> Scale
        scaled = matmul_qk/tf.math.sqrt(tf.cast(tf.shape(k)[-1], tf.float32))

        # Scale -> Mask
        scaled += (mask * -1e9) if mask is not None else 0

        # Mask -> Softmax
        attention = tf.nn.softmax(scaled, axis=-1)

        # Softmax, v -> MatMul
        retval = tf.matmul(attention, v)

        return retval, attention


    def split_heads(self, intra_tensor, batch_size):
        '''
        Split matrix for feeding into attention heads

        :param intra_tensor: matrix (q, k, or v)
        :param batch_size: batch size
        :return: split matrix
        '''

        intra_tensor = tf.reshape(intra_tensor, (batch_size, -1, self.num_heads, self.layer_width))
        return tf.transpose(intra_tensor, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask):
        # Feed forward and collect attention result and weights of attention network
        batch_size = tf.shape(q)[0]

        q = self.qlayer(q)
        k = self.klayer(k)
        v = self.vlayer(v)

        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        scaled, attention = self.scaled_dot_product(q, k, v, mask)
        scaled = tf.transpose(scaled, perm=[0, 2, 1, 3])
        concat = tf.reshape(scaled, (batch_size, -1, self.layer_width))

        output = self.dense(concat)

        return output, attention


class Encoder(tf.keras.layers.Layer):
    def __init__(self, layer_width, num_heads, feed_forward_params, rate):
        super(Encoder, self).__init__()

        self.multi_head = MultiHeadAttention(layer_width=layer_width, heads=num_heads)
        self.feed_forward = tf.keras.Sequential([
            tf.keras.layers.Dense(feed_forward_params, activation='relu'),
            tf.keras.layers.Dense(layer_width)
        ])

        self.multi_head_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.feed_forward_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout_attention = tf.keras.layers.Dropout(rate)
        self.dropout_feedforward = tf.keras.layers.Dropout(rate)

    def call(self, embedding, training, mask):
        attention, _ = self.multi_head(embedding, embedding, embedding, mask)
        attention = self.dropout_attention(attention, training=training)
        a_output = self.multi_head_norm(embedding + attention)

        f_output = self.feed_forward(a_output)
        f_output = self.dropout_feedforward(f_output, training=training)
        output = self.multi_head_norm(a_output + f_output)

        return output