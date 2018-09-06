import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import time
tfd = tf.contrib.distributions
tfb = tfd.bijectors
layers =  tf.contrib.layers

from bayesian_benchmarks.data import NYTaxi
data = NYTaxi()

DTYPE = tf.float32
NP_DTYPE = np.float32

# quite easy to interpret - multiplying by alpha causes a contraction in volume.
class LeakyReLU(tfb.Bijector):
    def __init__(self, alpha=0.5, validate_args=False, name="leaky_relu"):
        super(LeakyReLU, self).__init__(
            event_ndims=1, validate_args=validate_args, name=name)
        self.alpha = alpha

    def _forward(self, x):
        return tf.where(tf.greater_equal(x, 0), x, self.alpha * x)

    def _inverse(self, y):
        return tf.where(tf.greater_equal(y, 0), y, 1. / self.alpha * y)

    def _inverse_log_det_jacobian(self, y):
        event_dims = self._event_dims_tensor(y)
        I = tf.ones_like(y)
        J_inv = tf.where(tf.greater_equal(y, 0), I, 1.0 / self.alpha * I)
        # abs is actually redundant here, since this det Jacobian is > 0
        log_abs_det_J_inv = tf.log(tf.abs(J_inv))
        return tf.reduce_sum(log_abs_det_J_inv, axis=event_dims)


def net(x, out_size):
    return layers.stack(x, layers.fully_connected, [512, 512, out_size])


class NVPCoupling(tfb.Bijector):
    """
    NVP affine coupling layer for 2D units.
    """
    def __init__(self, D, d, layer_id=0, validate_args=False, name="NVPCoupling"):
        """
        Args:
          d: First d units are pass-thru units.
        """
        # first d numbers decide scaling/shift factor for remaining D-d numbers.
        super(NVPCoupling, self).__init__(
            event_ndims=1, validate_args=validate_args, name=name)
        self.D, self.d = D, d
        self.id = layer_id
        # create variables here
        tmp = tf.placeholder(dtype=DTYPE, shape=[1, self.d])
        self.s(tmp)
        self.t(tmp)

    def s(self, xd):
        with tf.variable_scope('s%d' % self.id, reuse=tf.AUTO_REUSE):
            return net(xd, self.D - self.d)

    def t(self, xd):
        with tf.variable_scope('t%d' % self.id, reuse=tf.AUTO_REUSE):
            return net(xd, self.D - self.d)

    def _forward(self, x):
        xd, xD = x[:, :self.d], x[:, self.d:]
        yD = xD * tf.exp(self.s(xd)) + self.t(xd)  # [batch, D-d]
        return tf.concat([xd, yD], axis=1)

    def _inverse(self, y):
        yd, yD = y[:, :self.d], y[:, self.d:]
        xD = (yD - self.t(yd)) * tf.exp(-self.s(yd))
        return tf.concat([yd, xD], axis=1)

    def _forward_log_det_jacobian(self, x):
        event_dims = self._event_dims_tensor(x)
        xd = x[:, :self.d]
        return tf.reduce_sum(self.s(xd), axis=event_dims)


class BatchNorm(tfb.Bijector):
    def __init__(self, eps=1e-5, decay=0.95, validate_args=False, name="batch_norm"):
        super(BatchNorm, self).__init__(
            event_ndims=1, validate_args=validate_args, name=name)
        self._vars_created = False
        self.eps = eps
        self.decay = decay

    def _create_vars(self, x):
        n = x.get_shape().as_list()[1]
        with tf.variable_scope(self.name):
            self.beta = tf.get_variable('beta', [1, n], dtype=DTYPE)
            self.gamma = tf.get_variable('gamma', [1, n], dtype=DTYPE)
            self.train_m = tf.get_variable(
                'mean', [1, n], dtype=DTYPE, trainable=False)
            self.train_v = tf.get_variable(
                'var', [1, n], dtype=DTYPE, initializer=tf.ones_initializer, trainable=False)
        self._vars_created = True

    def _forward(self, u):
        if not self._vars_created:
            self._create_vars(u)
        return (u - self.beta) * tf.exp(-self.gamma) * tf.sqrt(self.train_v + self.eps) + self.train_m

    def _inverse(self, x):
        # Eq 22. Called during training of a normalizing flow.
        if not self._vars_created:
            self._create_vars(x)
        # statistics of current minibatch
        m, v = tf.nn.moments(x, axes=[0], keep_dims=True)
        # update train statistics via exponential moving average
        update_train_m = tf.assign_sub(
            self.train_m, self.decay * (self.train_m - m))
        update_train_v = tf.assign_sub(
            self.train_v, self.decay * (self.train_v - v))
        # normalize using current minibatch statistics, followed by BN scale and shift
        with tf.control_dependencies([update_train_m, update_train_v]):
            return (x - m) * 1. / tf.sqrt(v + self.eps) * tf.exp(self.gamma) + self.beta

    def _inverse_log_det_jacobian(self, x):
        # at training time, the log_det_jacobian is computed from statistics of the
        # current minibatch.
        if not self._vars_created:
            self._create_vars(x)
        _, v = tf.nn.moments(x, axes=[0], keep_dims=True)
        abs_log_det_J_inv = tf.reduce_sum(
            self.gamma - .5 * tf.log(v + self.eps))
        return abs_log_det_J_inv

class Model:
    def __init__(self, X, batchsize=1000):
        data = tf.data.Dataset.from_tensor_slices(tf.identity(X))
        data = data.repeat()
        data_batch = data.batch(batch_size=batchsize)

        self._iterator_tensor = data_batch.make_initializable_iterator()
        self.X_train = self._iterator_tensor.get_next()
        self.X_test = tf.placeholder(DTYPE, [None, 2])

        flow = self.make_flow()
        base_dist = tfd.MultivariateNormalDiag(loc=tf.zeros([2], DTYPE))
        self.dist = tfd.TransformedDistribution(distribution=base_dist, bijector=flow)

        x = base_dist.sample(512)
        samples = [x]
        names = [base_dist.name]
        for bijector in reversed(self.dist.bijector.bijectors):
            x = bijector.forward(x)
            samples.append(x)
            names.append(bijector.name)
        self.names = names
        self.samples = samples

        self.logp_train = self.dist.log_prob(self.X_train)
        self.logp_test = self.dist.log_prob(self.X_test)
        self.loss = -tf.reduce_mean(self.logp_train)

        self.train_op = tf.train.AdamOptimizer(1e-3).minimize(self.loss)

    def make_flow(self):
        d, r = 2, 2
        bijectors = []
        num_layers = 6
        for i in range(num_layers):
            with tf.variable_scope('bijector_%d' % i):
                V = tf.get_variable('V', [d, r], dtype=DTYPE)  # factor loading
                shift = tf.get_variable('shift', [d], dtype=DTYPE)  # affine shift
                L = tf.get_variable('L', [d * (d + 1) / 2],
                                    dtype=DTYPE)  # lower triangular
                bijectors.append(tfb.Affine(
                    scale_tril=tfd.fill_triangular(L),
                    scale_perturb_factor=V,
                    shift=shift,
                ))
                alpha = tf.abs(tf.get_variable('alpha', [], dtype=DTYPE)) + .01
                bijectors.append(LeakyReLU(alpha=alpha))
        # Last layer is affine. Note that tfb.Chain takes a list of bijectors in the *reverse* order
        # that they are applied..
        mlp_bijector = tfb.Chain(
            list(reversed(bijectors[:-1])), name='2d_mlp_bijector')
        return mlp_bijector

    def plot_samples(self, sess):
        samples = sess.run(model.samples)
        f, arr = plt.subplots(1, len(samples), figsize=(4 * (len(samples)), 4))
        X0 = samples[0]


        arr[-1].scatter(data.X_train[::100, 0], data.X_train[::100, 1],
                        marker='.', color='yellow', alpha=0.3)

        for i in range(len(samples)):
            X1 = samples[i]
            idx = np.logical_and(X0[:, 0] < 0, X0[:, 1] < 0)
            arr[i].scatter(X1[idx, 0], X1[idx, 1], s=10, color='red')
            idx = np.logical_and(X0[:, 0] > 0, X0[:, 1] < 0)
            arr[i].scatter(X1[idx, 0], X1[idx, 1], s=10, color='green')
            idx = np.logical_and(X0[:, 0] < 0, X0[:, 1] > 0)
            arr[i].scatter(X1[idx, 0], X1[idx, 1], s=10, color='blue')
            idx = np.logical_and(X0[:, 0] > 0, X0[:, 1] > 0)
            arr[i].scatter(X1[idx, 0], X1[idx, 1], s=10, color='black')
            # arr[i].set_xlim([-2, 2])
            # arr[i].set_ylim([-2, 2])
            arr[i].set_title(self.names[i])

        plt.show()


USE_BATCHNORM = True
num_bijectors = 8

class FancyModel(Model):
    def __init__(self, model_type, *args, **kwargs):
        self.model_type = model_type
        Model.__init__(self, *args, **kwargs)

    def make_flow(self):
        bijectors = []

        for i in range(num_bijectors):
            if self.model_type == 'NVP':
                bijectors.append(NVPCoupling(D=2, d=1, layer_id=i))

            elif self.model_type == 'MAF':
                bijectors.append(tfb.MaskedAutoregressiveFlow(
                    shift_and_log_scale_fn=tfb.masked_autoregressive_default_template(
                        hidden_layers=[512, 512])))

            elif self.model_type == 'IAF':
                bijectors.append(tfb.Invert(tfb.MaskedAutoregressiveFlow(
                    shift_and_log_scale_fn=tfb.masked_autoregressive_default_template(
                        hidden_layers=[512, 512]))))

            if USE_BATCHNORM and i % 2 == 0:
                # BatchNorm helps to stabilize deep normalizing flows, esp. Real-NVP
                bijectors.append(BatchNorm(name='batch_norm%d' % i))

            bijectors.append(tfb.Permute(permutation=[1, 0]))

        # Discard the last Permute layer.
        return tfb.Chain(list(reversed(bijectors[:-1])))





sess = tf.Session()
model = FancyModel('IAF', data.X_train[:, :2].astype(np.float32),
                   batchsize=5000)
sess.run(tf.global_variables_initializer())
sess.run(model._iterator_tensor.initializer)

L = []
plot_freq = 100



t = time.time()

try:
    for it in range(int(1e4)):
        _, loss = sess.run([model.train_op, model.loss])#, {model.X_train : X})
        L.append(loss)
        if it % plot_freq == 0:
            print('{} {}'.format(it, loss))
except KeyboardInterrupt:
    pass

print('train time {:.4f}s'.format(time.time() - t))

# plt.plot(L)
# plt.show()

model.plot_samples(sess)


# plt.axes().set_xlim(-74.1, -73.7)
# plt.axes().set_ylim(40.6, 40.9)



# x2_dist = tfd.Normal(loc=0., scale=4.)
# x2_samples = x2_dist.sample(batch_size)
# x1 = tfd.Normal(loc=.25 * tf.square(x2_samples),
#                 scale=tf.ones(batch_size, dtype=tf.float32))
# x1_samples = x1.sample()
# x_samples = tf.stack([x1_samples, x2_samples], axis=1)
#
#
#
# x_placeholder = tf.placeholder(tf.float32, [None, 2])
# logp = dist.log_prob(x_placeholder)
#
#

#
#
# # results = sess.run(samples)

# #
#
#
# NUM_STEPS = int(3e5)
# global_step = []
# np_losses = []
# for i in range(NUM_STEPS):
#     _, np_loss = sess.run([train_op, loss])
#     if i % 1000 == 0:
#         global_step.append(i)
#         np_losses.append(np_loss)
#     if i % int(1e4) == 0:
#         print(i, np_loss)
# start = 10
# plt.plot(np_losses[start:])
# plt.show()
#
#
# # results = sess.run(samples)
# # f, arr = plt.subplots(1, len(results), figsize=(4 * (len(results)), 4))
# # X0 = results[0]
# # for i in range(len(results)):
# #     X1 = results[i]
# #     idx = np.logical_and(X0[:, 0] < 0, X0[:, 1] < 0)
# #     arr[i].scatter(X1[idx, 0], X1[idx, 1], s=10, color='red')
# #     idx = np.logical_and(X0[:, 0] > 0, X0[:, 1] < 0)
# #     arr[i].scatter(X1[idx, 0], X1[idx, 1], s=10, color='green')
# #     idx = np.logical_and(X0[:, 0] < 0, X0[:, 1] > 0)
# #     arr[i].scatter(X1[idx, 0], X1[idx, 1], s=10, color='blue')
# #     idx = np.logical_and(X0[:, 0] > 0, X0[:, 1] > 0)
# #     arr[i].scatter(X1[idx, 0], X1[idx, 1], s=10, color='black')
# #     # arr[i].set_xlim([-2, 2])
# #     # arr[i].set_ylim([-2, 2])
# #     arr[i].set_title(names[i])
# # plt.show()
#
#
# X1 = sess.run(dist.sample(4000))
# plt.scatter(X1[:, 0], X1[:, 1], color='red', s=2)
# # arr[i].set_ylim([-.5, .5])
# plt.show()
#
#
# # x_samples_np = sess.run([x1_samples, x2_samples])
# # plt.scatter(*x_samples_np)
# # plt.show()
#
# N = 100
# l = np.linspace(-5, 5, N)
# X_grid = np.array([x.flatten() for x in np.meshgrid(l, l)]).T
#
#
# lp = sess.run(logp, {x_placeholder:X_grid})
#
# plt.pcolor(np.exp(lp).reshape(N, N))
# plt.colorbar()
# plt.show()
#
# print(np.sum(np.exp(lp)))
#
