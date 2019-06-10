import tensorflow as tf
from decoder import Decoder
from encoder import Encoder
from prior import Prior
from tensorflow.examples.tutorials.mnist import input_data
from plot import Plot
import matplotlib.pyplot as plt

tfd = tf.contrib.distributions


def fillPlot(draw_data, ax, mnist):
    feed = {data: mnist.test.images.reshape([-1, 28, 28])}
    test_elbo, test_codes, test_samples = sess.run(draw_data, feed)

    print('Epoch', epoch, 'elbo', test_elbo)
    ax[epoch, 0].set_ylabel('Epoch {}'.format(epoch))
    Plot.plot_codes(ax[epoch, 0], test_codes, mnist.test.labels)
    Plot.plot_samples(ax[epoch, 1:], test_samples)


make_encoder = tf.make_template('encoder', Encoder.make_encoder)
prior = Prior.make_prior(code_size=2)
make_decoder = tf.make_template('decoder', Decoder.make_decoder)

########################
# Define loss
data = tf.placeholder(tf.float32, [None, 28, 28])
posterior = make_encoder(data, code_size=2)
code = posterior.sample()

# ln p(x|z) - логарифмическая вероятность для кодировщика
likelihood = make_decoder(code, [28, 28]).log_prob(data)
# отклонение. Использует расхождение Кульбака - Лейблера
divergence = tfd.kl_divergence(posterior, prior)
# evidence lower bound (ELBO) - нижняя граница доказательств
# вычисляет среднее
elbo = tf.reduce_mean(likelihood - divergence)
########################
optimize = tf.train.AdamOptimizer(0.001).minimize(-elbo)

######
sampleAmount = 5
trainAmount = 20
exampleAmount = 2

samples = make_decoder(prior.sample(sampleAmount), [28, 28]).mean()

mnist = input_data.read_data_sets('MNIST_data/')
fig, ax = plt.subplots(nrows=exampleAmount, ncols=sampleAmount + 1, figsize=(sampleAmount, exampleAmount))
with tf.train.MonitoredSession() as sess:
    for epoch in range(exampleAmount):
        draw_data = [elbo, code, samples]
        fillPlot(draw_data, ax, mnist)
        for _ in range(trainAmount):
            feed = {data: mnist.train.next_batch(100)[0].reshape([-1, 28, 28])}
            sess.run(optimize, feed)

plt.savefig('vae-mnist.png', dpi=300, transparent=True, bbox_inches='tight')
