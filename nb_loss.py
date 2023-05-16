import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow import keras

# Minor modifications with the addidion of the term (1e-30) as problems arise when n = 0, which can be caused by the eventdata being 0
# ReLu(k) is used as problems occur when k= -n, as -inf + inf = nan
def log_pmf(k, n, p):
    gamma_binomial = tf.math.lgamma(tf.nn.relu(k) + n + 1e-30) - tf.math.lgamma(n + 1e-30) - tf.math.lgamma(k + 1.0)
    return gamma_binomial + n * tf.math.log(p) + k * tf.math.log(1.0 - p)


# convolution pmf utilizes count_up_vector and count_down_matrix to calculate the convolution logpmf. 
# Irrelevant values for each vector will cancel each other because logpmf of a negative value is -inf
# The exponential in turn gives zero. 
def convolution_pmf(t, n0, n1, p0, p1):
    t_max = tf.reduce_sum(t)
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
