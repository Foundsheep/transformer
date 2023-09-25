from abc import ABC

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import numpy as np
import time


class EmbeddingLayer(tf.keras.layers.Layer):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.emb = tf.keras.layers.Embedding(input_dim=self.vocab_size, output_dim=self.d_model, dtype=tf.float32)

        # TODO : sums of embeddings, positional encoding -> dropout needed

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
        pos = pos.numpy()  # to numpy because tensor doesn't support direct assignment
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
        x = tf.py_function(func=self.add_positional_encoding, inp=[x, True], Tout=tf.float32)
        # x = self.add_positional_encoding(x, is_plot=True)
        print(f"=== Embedding done, shape : [{x.shape}]")
        return x


class Attention(tf.keras.layers.Layer):
    def __init__(self, is_masking=False):
        super().__init__()
        self.is_masking = is_masking

    def masking(self, x):
        start = time.time()
        assert len(x.shape) == 3
        result = np.ones_like(x)
        for b_idx, b in enumerate(x):
            sequence_length = len(b)
            for i, line in enumerate(b):
                if i != sequence_length - 1:
                    result[b_idx, i, i+1:] = -np.inf
                    result[b_idx, i, :i+1] = line[:i+1]
                else:
                    result[b_idx, i, :] = line
        result = tf.convert_to_tensor(result)
        end = time.time()
        print(f"===masking done, shape: [{result.shape}], time took [{end - start :.3f}] seconds")
        return result

    def call(self, Q, K, V):
        x = tf.linalg.matmul(Q, K, transpose_b=True)
        x = tf.math.divide(x, tf.math.sqrt(float(Q.shape[-1])))
        if self.is_masking:
            x = self.masking(x)
        x = tf.nn.softmax(x)
        x = tf.linalg.matmul(x, V)
        print(f"=== Attention done, shape : [{x.shape}]")
        return x


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, h, d_k, d_v, is_masking=False):
        super().__init__()
        self.d_k = d_k
        self.d_v = d_v
        self.d_model = d_model
        self.h = h
        self.is_masking = is_masking
        self.linear_layer_Q_list = [tf.keras.layers.Dense(self.d_k) for _ in range(h)]
        self.linear_layer_K_list = [tf.keras.layers.Dense(self.d_k) for _ in range(h)]
        self.linear_layer_V_list = [tf.keras.layers.Dense(self.d_v) for _ in range(h)]
        self.attention_list = [Attention(is_masking=self.is_masking) for _ in range(h)]
        self.concat = tf.keras.layers.Concatenate()
        self.linear_layer_end = tf.keras.layers.Dense(units=self.d_model)
        self.drop_out = tf.keras.layers.Dropout(rate=0.1)

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
        # assert len(inputs.shape) == 3
        batch, length, = 64, 52  # TODO : hard-carding to be modified
        new_channel = int(self.d_model / self.h)
        print(f"=== new_channel : [{new_channel}]")
        x = tf.reshape(inputs, shape=(batch, -1, length, new_channel))
        # x = tf.transpose(x, perm=(0, 2, 1, 3))
        print(F"=== dividing by head number... shape: [{x.shape}]")
        return x

    def call(self, Q, K, V):
        Q_reshaped = self.reshape_tensors(Q)
        K_reshaped = self.reshape_tensors(K)
        V_reshaped = self.reshape_tensors(V)
        Q_projected_list = [self.linear_layer_Q_list[i](Q_reshaped[:, i, :, :]) for i in range(self.h)]
        K_projected_list = [self.linear_layer_K_list[i](K_reshaped[:, i, :, :]) for i in range(self.h)]
        V_projected_list = [self.linear_layer_V_list[i](V_reshaped[:, i, :, :]) for i in range(self.h)]
        print(f"Q_projected shape : [{Q_projected_list[0].shape}]")
        print(f"K_projected shape : [{K_projected_list[0].shape}]")
        print(f"V_projected shape : [{V_projected_list[0].shape}]")
        attention_result_list = [self.attention_list[i](Q_projected_list[i], K_projected_list[i], V_projected_list[i]) for i in range(self.h)]
        print(f"attention_list element shape: [{attention_result_list[0].shape}]")
        x = self.concat(attention_result_list)
        print(f"=== after reshaping : [{x.shape}]")
        x = self.linear_layer_end(x)
        x = self.drop_out(x)
        print(f"=== Multi-Head attention done, shape : [{x.shape}]")
        return x


class FeedFowardNetwork(tf.keras.layers.Layer):
    def __init__(self, d_model, d_ff=2048):
        super().__init__()
        self.d_model = d_model
        self.linear_1 = tf.keras.layers.Dense(d_ff, use_bias=True)
        self.linear_2 = tf.keras.layers.Dense(d_model, use_bias=True)
        self.relu = tf.keras.layers.ReLU()
        self.drop_out = tf.keras.layers.Dropout(rate=0.1)

    def call(self, inputs):
        x = self.linear_1(inputs)
        x = self.relu(x)
        x = self.linear_2(x)
        x = self.drop_out(x)
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
        self.d_model = d_model
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
    def __init__(self, d_k, d_v, d_model, h, is_masking):
        super().__init__()
        self.d_k = d_k
        self.d_v = d_v
        self.d_model = d_model
        self.h = h
        self.is_masking = is_masking
        self.add_layer = tf.keras.layers.Add()
        self.MHA_1 = MultiHeadAttention(d_k=self.d_k, d_v=self.d_v, d_model=self.d_model, h=self.h, is_masking=self.is_masking)
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
    def __init__(self, N, d_k, d_v, d_model, h, is_masking):
        super().__init__()
        self.N = N
        self.d_k = d_k
        self.d_v = d_v
        self.d_model = d_model
        self.h = h
        self.is_masking = is_masking
        self.decoder_dict = {}
        for i in range(self.N):
            self.decoder_dict[f"decoder_block_{i}"] = DecoderBlock(d_k=self.d_k, d_v=self.d_v, d_model=self.d_model, h=self.h, is_masking=self.is_masking)

    def call(self, Q, K, V, *args, **kwargs):
        x = Q
        for i in range(self.N):
            x = self.decoder_dict[f"decoder_block_{i}"](x, K, V)
        return x


class Transformer(tf.keras.Model, ABC):
    def __init__(self, d_k, d_v, d_model, vocab_size, h, N):
        super().__init__()
        self.d_k = d_k
        self.d_v = d_v
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.h = h
        self.N = N
        self.enc_embedding_layer = EmbeddingLayer(d_model=self.d_model, vocab_size=self.vocab_size)
        self.dec_embedding_layer = EmbeddingLayer(d_model=self.d_model, vocab_size=self.vocab_size)
        self.encoder = Encoder(d_k=self.d_k, d_v=self.d_v, d_model=self.d_model, h=self.h, N=self.N)
        self.decoder = Decoder(d_k=self.d_k, d_v=self.d_v, d_model=self.d_model, h=self.h, N=self.N, is_masking=True)
        self.last_linear = tf.keras.layers.Dense(self.vocab_size)
        self.last_softmax = tf.keras.layers.Softmax()

    def call(self, inputs, training=True, mask=None):
        input_Q, input_K, input_V = inputs[0][0], inputs[0][1], inputs[1]
        if training:
            emb_Q = self.enc_embedding_layer(input_Q)
            emb_K = self.enc_embedding_layer(input_K)
            emb_V = self.enc_embedding_layer(input_V)
            emb_dec = self.dec_embedding_layer(input_V)
            enc_output = self.encoder(Q=emb_Q, K=emb_K, V=emb_V)
            dec_output = self.decoder(Q=emb_dec, K=enc_output, V=enc_output)
            outputs = self.last_linear(dec_output)
            outputs = self.last_softmax(outputs)
            return outputs
        else:
            emb_Q = self.enc_embedding_layer(input_Q)
            emb_K = self.enc_embedding_layer(input_K)
            emb_V = self.enc_embedding_layer(input_V)
            emb_dec = self.dec_embedding_layer(ipnut_V)
            enc_output = self.encoder(Q=emb_Q, K=emb_K, V=emb_V)
            dec_output = self.decoder(Q=emb_dec, K=enc_output, V=enc_output)
            outputs = self.last_linear(dec_output)
            outputs = self.last_softmax(outputs)
            return outputs


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super().__init__()
        self.d_model = tf.cast(d_model, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        step = tf.cast(step, dtype=tf.float32)
        elm_1 = step ** -0.5
        elm_2 = step * (self.warmup_steps ** -1.5)
        return (self.d_model ** -0.5) * tf.math.minimum(elm_1, elm_2)


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
    Q = tf.random.uniform(shape=(b_size, s_size), maxval=512, dtype=tf.int32)
    K = tf.random.uniform(shape=(b_size, s_size), maxval=512, dtype=tf.int32)
    V = tf.random.uniform(shape=(b_size, s_size), maxval=512, dtype=tf.int32)

    # # layers
    # Q_emb = EmbeddingLayer(vocab_size=vocab_size, d_model=d_model)(Q)
    # K_emb = EmbeddingLayer(vocab_size=vocab_size, d_model=d_model)(K)
    # V_emb = EmbeddingLayer(vocab_size=vocab_size, d_model=d_model)(V)
    #
    # output = MultiHeadAttention(d_model=d_model, h=h, d_k=d_k, d_v=d_v)(Q_emb, K_emb, V_emb)
    # print("output:", output.shape)
    # print("================================================")
    #
    # output_2 = Encoder(N=N, d_k=d_k, d_v=d_v, d_model=d_model, h=h)(Q_emb, K_emb, V_emb)
    # print("output_2:", output_2.shape)
    # print("================================================")
    #
    # output_3 = Decoder(N=N, d_k=d_k, d_v=d_v, d_model=d_model, h=h, is_masking=True)(Q_emb, K_emb, V_emb)
    # print("output_3", output_3.shape)

    # TODO : adam optimizer
    # TODO : dropout
    transformer = Transformer(d_k=d_k, d_v=d_v, d_model=d_model, vocab_size=vocab_size, h=h, N=N)
    transformer.compile(optimizer="adam", loss="cross_entropy", run_eagerly=True)
    transformer.fit((Q, K), epochs=5)


