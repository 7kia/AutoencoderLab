import tensorflow as tf
import numpy as np

tfd = tf.contrib.distributions


class Decoder:
    @staticmethod
    def make_decoder(code, data_shape):
        x = code
        x = tf.layers.dense(x, 200, tf.nn.relu)
        x = tf.layers.dense(x, 200, tf.nn.relu)
        logit = tf.layers.dense(x, np.prod(data_shape))
        logit = tf.reshape(logit, [-1] + data_shape)
        return tfd.Independent(tfd.Bernoulli(logit), 2)
