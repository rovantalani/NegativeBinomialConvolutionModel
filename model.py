import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import glob
import math
import tensorflow as tf
from tensorflow import keras


class NegativeBinomialLayer(keras.layers.Layer):
    def __init__(self, r_size=2, p_size=2, input_vector=None):
        super(NegativeBinomialLayer, self).__init__()

        log_r_init = tf.constant(0.0, shape=r_size, dtype=tf.float32)
        self.log_r = tf.Variable(initial_value=log_r_init, trainable=True)

        logit_p_init = tf.constant(0.0, shape=p_size, dtype=tf.float32)
        self.logit_p = tf.Variable(initial_value=logit_p_init, trainable=True)

        if input_vector is None:
            partition_vector_init = tf.constant(tf.random.normal([r_size], stddev=0.1), dtype=tf.float32)
            self.partition_vector = tf.Variable(initial_value=partition_vector_init, trainable=True)

        else:
            partition_vector_init = tf.constant(input_vector, dtype=tf.float32)
            self.partition_vector = tf.Variable(initial_value=partition_vector_init, trainable=False)

    def get_r(self):
        return tf.math.exp(self.log_r)

    def get_p(self):
        return tf.math.sigmoid(self.logit_p)

    def get_pv1(self):
        return tf.math.sigmoid(self.partition_vector)

    def get_pv0(self):
        return tf.math.sigmoid(-self.partition_vector)

    def flip(self):
        self.partition_vector = -self.partition_vector
        self.logit_p = tf.reverse(self.logit_p, axis=[0])

    def call(self, inputs):

        inputs_times_r = inputs * self.get_r()
        inputs_type0 = tf.reduce_sum(inputs_times_r * self.get_pv0(), axis=1, keepdims=True)
        inputs_type1 = tf.reduce_sum(inputs_times_r * self.get_pv1(), axis=1, keepdims=True)

        n = tf.concat([inputs_type0, inputs_type1], axis=1)
        p_ones = tf.ones(shape=[inputs.shape[0], 1])
        p = p_ones * self.get_p()

        return tf.stack([n, p], axis=2)

def log_pmf(k, n, p):
    gamma_binomial = tf.math.lgamma(tf.nn.relu(k) + n + 1e-30) - tf.math.lgamma(n + 1e-30) - tf.math.lgamma(k + 1.0)
    return gamma_binomial + n * tf.math.log(p) + k * tf.math.log(1.0 - p)


def convolution_pmf(t, n0, n1, p0, p1):
    t_max = tf.reduce_sum(t)
    print(t_max)
    count_up_vector = tf.reshape(tf.cast(tf.range(t_max + 1), tf.float32), shape=[1, -1])
    count_down_matrix = t - count_up_vector
    return tf.reduce_sum(tf.math.exp(log_pmf(count_up_vector, n0, p0) + log_pmf(count_down_matrix, n1, p1)), axis=1)


def my_pmf(y_true, y_pred):
    n, p = tf.unstack(y_pred, axis=2)
    n0 = n[:, 0:1]
    n1 = n[:, 1:2]
    p0 = p[:, 0:1]
    p1 = p[:, 1:2]
    return convolution_pmf(y_true, n0, n1, p0, p1)


def my_loss_fn(y_true, y_pred):
    return -tf.reduce_mean(tf.math.log(my_pmf(y_true, y_pred)))
