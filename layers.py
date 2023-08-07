import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


class EmbeddingLayer(tf.keras.layers.Layer):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.emb = tf.keras.layers.Embedding(input_dim=self.vocab_size, output_dim=self.d_model, dtype=tf.float32)
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

    def call(self, inputs):
        x = self.emb(inputs)
        x = self.add_positional_encoding(x, is_plot=True)
        print(f"=== Embedding done, shape : [{x.shape}]")
        return x

    def masking(self):
        # TODO : masking function in decoder
        pass


class Attention(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__()

    def call(self, Q, K, V):
        x = tf.linalg.matmul(Q, K, transpose_b=True)
        x = tf.math.divide(x, tf.math.sqrt(float(Q.shape[-1])))
        x = tf.nn.softmax(x)
        x = tf.linalg.matmul(x, V)
        print(f"=== Attention done, shape : [{x.shape}]")
        return x


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, h, d_k, d_v):
        super().__init__()
        self.d_k = d_k
        self.d_v = d_v
        self.d_model = d_model
        self.h = h
        self.linear_layer_Q = tf.keras.layers.Dense(self.d_k)
        self.linear_layer_K = tf.keras.layers.Dense(self.d_k)
        self.linear_layer_V = tf.keras.layers.Dense(self.d_v)
        self.attn = Attention()
        # self.concat = tf.keras.layers.concatenate()
        self.linear_layer_end = tf.keras.layers.Dense(units=self.d_model)

    def split_tensors(self, inputs, h):
        assert len(inputs.shape) == 3
        batch, length, self.d_model = inputs.shape
        inputs = tf.expand_dims(inputs, 1)
        assert len(inputs.shape) == 4
        temp_list = []
        new_channel = int(d_model / h)
        for idx in range(h):
            temp_list.append(inputs[:, :, :, idx * new_channel:(idx + 1) * new_channel])
        inputs = tf.concat(temp_list, axis=1)
        return inputs

    def reshape_tensors(self, inputs):
        assert len(inputs.shape) == 3
        batch, length, = inputs.shape[0], inputs.shape[1]
        new_channel = int(self.d_model / self.h)
        print(f"=== new_channel : [{new_channel}]")
        x = tf.reshape(inputs, shape=(batch, length, -1, new_channel))
        # x = tf.transpose(x, perm=(0, 2, 1, 3))
        return x

    def call(self, Q, K, V):
        Q_reshaped = self.reshape_tensors(Q)
        K_reshaped = self.reshape_tensors(K)
        V_reshaped = self.reshape_tensors(V)
        Q_ = self.linear_layer_Q(Q_reshaped)
        K_ = self.linear_layer_K(K_reshaped)
        V_ = self.linear_layer_V(V_reshaped)
        print(f"Q_ shape : [{Q_.shape}")
        print(f"K_ shape : [{K_.shape}")
        print(f"V_ shape : [{V_.shape}")
        x = self.attn(Q_, K_, V_)
        batch, legnth = Q.shape[0], Q.shape[1]
        x = tf.reshape(x, shape=(batch, legnth, -1))
        x = self.linear_layer_end(x)
        print(f"=== Multi-Head attention done, shape : [{x.shape}]")
        return x


class FeedFowardNetwork(tf.keras.layers.Layer):
    def __init__(self, d_model, d_ff=2048):
        super().__init__()
        self.d_model = d_model
        self.linear_1 = tf.keras.layers.Dense(d_ff, use_bias=True)
        self.linear_2 = tf.keras.layers.Dense(d_model, use_bias=True)
        self.relu = tf.keras.layers.ReLU()

    def call(self, inputs):
        x = self.linear_1(inputs)
        x = self.relu(x)
        x = self.linear_2(x)
        return x


class EncoderBlock(tf.keras.layers.Layer):
    def __init__(self, d_k, d_v, d_model, h):
        super().__init__()
        self.d_k = d_k
        self.d_v = d_v
        self.d_model = d_model
        self.h = h
        self.add_layer = tf.keras.layers.Add()
        self.MHA = MultiHeadAttention(d_k=self.d_k, d_v=self.d_v, d_model=self.d_model, h=self.h)
        self.LN_1 = tf.keras.layers.LayerNormalization()
        self.LN_2 = tf.keras.layers.LayerNormalization()
        self.FF = FeedFowardNetwork(self.d_model)

    def call(self, Q, K, V, *args, **kwargs):
        x_ = Q
        x = self.MHA(Q, K, V)
        x = self.add_layer([x, x_])
        x_ = self.LN_1(x)
        x = self.FF(x_)
        x = self.add_layer([x, x_])
        x = self.LN_2(x)
        print(f"=== EncoderBlock Done, shape: [{x.shape}]")
        return x


class Encoder(tf.keras.layers.Layer):
    def __init__(self, N, d_k, d_v, d_model, h):
        super().__init__()
        self.N = N
        self.d_k = d_k
        self.d_v = d_v
        self.d_model =d_model
        self.h = h
        self.encoder_dict = {}
        for i in range(self.N):
            self.encoder_dict[f"encoder_block_{i}"] = EncoderBlock(d_k=self.d_k, d_v=self.d_v, d_model=self.d_model, h=self.h)

    def call(self, Q, K, V, *args, **kwargs):
        x = Q
        for i in range(self.N):
            x = self.encoder_dict[f"encoder_block_{i}"](x, K, V)
        return x


class DecoderBlock(tf.keras.layers.Layer):
    def __init__(self, d_k, d_v, d_model, h):
        super().__init__()
        self.d_k = d_k
        self.d_v = d_v
        self.d_model = d_model
        self.h = h
        self.add_layer = tf.keras.layers.Add()
        self.MHA_1 = MultiHeadAttention(d_k=self.d_k, d_v=self.d_v, d_model=self.d_model, h=self.h)
        self.MHA_2 = MultiHeadAttention(d_k=self.d_k, d_v=self.d_v, d_model=self.d_model, h=self.h)
        self.LN_1 = tf.keras.layers.LayerNormalization()
        self.LN_2 = tf.keras.layers.LayerNormalization()
        self.LN_3 = tf.keras.layers.LayerNormalization()
        self.FF = FeedFowardNetwork(self.d_model)

    def call(self, Q, K, V):
        x_ = Q
        x = self.MHA_1(Q, Q, Q)
        x = self.add_layer([x, x_])
        x = self.LN_1(x)
        x_ = x
        x = self.MHA_2(x, K, V)
        x = self.add_layer([x, x_])
        x = self.LN_2(x)
        x_ = x
        x = self.FF(x)
        x = self.add_layer([x, x_])
        x = self.LN_3(x)
        return x


class Decoder(tf.keras.layers.Layer):
    def __init__(self, N, d_k, d_v, d_model, h):
        super().__init__()
        self.N = N
        self.d_k = d_k
        self.d_v = d_v
        self.d_model =d_model
        self.h = h
        self.decoder_dict = {}
        for i in range(self.N):
            self.decoder_dict[f"decoder_block_{i}"] = DecoderBlock(d_k=self.d_k, d_v=self.d_v, d_model=self.d_model, h=self.h)

    def call(self, Q, K, V, *args, **kwargs):
        x = Q
        for i in range(self.N):
            x = self.decoder_dict[f"decoder_block_{i}"](x, K, V)
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

    # initialise hyper-parameters
    d_k = 64
    d_v = 64
    b_size = 64
    s_size = 100
    vocab_size = 10000
    d_model = 512
    h = 8
    N = 6

    # set the values
    Q = tf.random.uniform(shape=(b_size, s_size), maxval=1000, dtype=tf.int32)
    K = tf.random.uniform(shape=(b_size, s_size), maxval=1000, dtype=tf.int32)
    V = tf.random.uniform(shape=(b_size, s_size), maxval=1000, dtype=tf.int32)

    # layers
    Q_emb = EmbeddingLayer(vocab_size=vocab_size, d_model=d_model)(Q)
    K_emb = EmbeddingLayer(vocab_size=vocab_size, d_model=d_model)(K)
    V_emb = EmbeddingLayer(vocab_size=vocab_size, d_model=d_model)(V)

    output = MultiHeadAttention(d_model=d_model, h=h, d_k=d_k, d_v=d_v)(Q_emb, K_emb, V_emb)
    print("output:", output.shape)
    print("================================================")

    output_2 = Encoder(N=N, d_k=d_k, d_v=d_v, d_model=d_model, h=h)(Q_emb, K_emb, V_emb)
    print("output_2:", output_2.shape)
    print("================================================")

    output_3 = Decoder(N=N, d_k=d_k, d_v=d_v, d_model=d_model, h=h)(Q_emb, K_emb, V_emb)
    print("output_3", output_3.shape)


