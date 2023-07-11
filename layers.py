import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


class EmbeddingLayer(tf.keras.layers.Layer):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.d_model = d_model
        self.emb = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=d_model)
        pass

    def add_positional_encoding(self, x, is_plot=True):
        """
        Sinusoidal Positional_Encoding from Attention Is All You Need
        :param x: Input to positional encoding, which is output of initial embedding layer
        :param is_plot: Save the positional encoding plot to check
        :return: tensor in the same shape with the input after adding the positional encoding value
        """
        B, S, D = x.shape
        x_axis = np.arange(S)[:, np.newaxis] # (seq, 1)
        y_axis = np.arange(int(D / 2))[np.newaxis, :] * 2 # (1, d_model/2)
        y_axis = tf.repeat(y_axis, 2, axis=1) # (1, d_model)

        # just in case d_model is odd
        if D % 2 == 1:
            to_concat = tf.cast(tf.reshape(D, shape=(-1, 1)), dtype=y_axis.dtype)
            y_axis = tf.concat([y_axis, to_concat], axis=1)

        # calculate the core part
        pos = x_axis / (10000 ** (y_axis / D))
        pos = pos.numpy() # to numpy because tensor doesn't support direct assignment
        pos[:, 0::2] = np.sin(pos[:, 0::2])
        pos[:, 1::2] = np.cos(pos[:, 1::2])

        # see if the positional encoding has been created correctly
        if is_plot:
            plt.pcolormesh(pos)
            plt.colorbar()
            plt.ylabel('Position')
            plt.xlabel('Depth')
            plt.savefig('./positional_encoding.jpg')

        return tf.add(x, pos)

    def call(self, x):
        return self.emb(x)

    def masking(self):
        # TODO : masking function in decoder
        pass


class Attention(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()

        # # make sure Q, K, V are all vectors in batches and Q, K have the same shape
        # assert len(Q.shape) == 2 and len(K.shape) == 2 and len(V.shape) == 2
        # assert Q.shape[-1] == K.shape[-1]
        # self.Q = Q
        # self.K = K
        # self.V = V

    def call(self, inputs):
        Q, K, V = inputs
        x = tf.linalg.matmul(Q, K, transpose_b=True)
        x = tf.math.divide(x, tf.math.sqrt(Q.shape[-1]))
        x = tf.nn.softmax(x)
        x = tf.linalg.matmul(x, V)
        return x


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_k, d_v, d_model, h):
        super().__init__()
        assert int(d_model / h) == d_k == d_v
        self.linear_layer_dict = dict()
        for i in range(h):
            self.linear_layer_dict[f"Q_{i}"] = tf.keras.layers.Dense(units=d_k)
            self.linear_layer_dict[f"K_{i}"] = tf.keras.layers.Dense(units=d_k)
            self.linear_layer_dict[f"V_{i}"] = tf.keras.layers.Dense(units=d_v)
        self.attn = Attention()
        self.concat = tf.keras.layers.concatenate()
        self.linear_layer_end = tf.keras.layers.Dense(units=d_model)
        self.h = h

    def split_tensors(self, inputs, h):
        assert len(inputs.shape) == 3
        batch, length, d_model = inputs.shape
        inputs = tf.expand_dims(inputs, 1)
        assert len(inputs.shape) == 4
        temp_list = []
        new_channel = int(d_model / h)
        for idx in range(h):
            temp_list.append(inputs[:, :, :, idx * new_channel:(idx + 1) * new_channel])
        inputs = tf.concat(temp_list, axis=1)
        return inputs

    def call(self, inputs):
        _Q, _K, _V = inputs
        Q_split = self.split_tensors(_Q, self.h)
        K_split = self.split_tensors(_K, self.h)
        V_split = self.split_tensors(_V, self.h)

        # for i in range(self.h):
        #     Q_split[:, i, :, :]

        x = self.attn(Q, K, V)

        # for loop(?) and concat afterwards
        x = self.linear_layer_end(x)
        return x


def test_class(C, b_size, s_size, **kwargs):
    c = C(**kwargs)

    # Expects to receive (batch_size, sequence_size) shape
    x = tf.random.uniform(shape=(b_size, s_size), maxval=1000, dtype=tf.int32)
    print(f"test case original shape : {x.numpy().shape}")
    x = c(x)
    print(f"test case afterwards shape : {x.numpy().shape}")
    print(f"type of the return value : {type(x)}")

    x = c.add_positional_encoding(x, is_plot=True)
    print(f"After adding positional encoding : {x.numpy().shape}")

if __name__ == '__main__':
    test_class(EmbeddingLayer, b_size=64, s_size=100, vocab_size=10000, d_model=512)
