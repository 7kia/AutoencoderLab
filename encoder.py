import tensorflow as tf

tfd = tf.contrib.distributions


# q (z | x)
class Encoder:
    @staticmethod
    def make_encoder(data, code_size):
        x = tf.layers.flatten(data)
        x = tf.layers.dense(x, 200, tf.nn.relu)
        x = tf.layers.dense(x, 200, tf.nn.relu)
        loc = tf.layers.dense(x, code_size)
        scale = tf.layers.dense(x, code_size, tf.nn.softplus)
        return tfd.MultivariateNormalDiag(loc, scale)
