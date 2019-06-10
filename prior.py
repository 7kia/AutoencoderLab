import tensorflow as tf

tfd = tf.contrib.distributions


class Prior:
    # p (z)
    @staticmethod
    def make_prior(code_size):
        # нормальное распределение с нулевым средним / zero mean
        loc = tf.zeros(code_size)
        # c единичной дисперсией
        scale = tf.ones(code_size)
        # TODO
        return tfd.MultivariateNormalDiag(loc, scale)
