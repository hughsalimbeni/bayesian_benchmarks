from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import math
import sympy as sp

from gpflow.transforms import positive
from gpflow.params import Parameter, Parameterized
from gpflow.kernels import Kernel, Static
from gpflow import params_as_tensors
import gpflow.kernels as gfk
from gpflow import settings
float_type = settings.float_type


class _KernelLayer(Parameterized):
    def __init__(self, input_dim, name):
        Parameterized.__init__(self, name=name)
        self.input_dim = input_dim

    def forward(self, input):
        raise NotImplementedError

    def symbolic(self, ks):
        raise NotImplementedError


class Linear(_KernelLayer):
    def __init__(self, input_dim, output_dim, name):
        super(Linear, self).__init__(input_dim, name)
        self.output_dim = output_dim

        min_w, max_w = 1. / (2 * self.input_dim), 3. / (2 * self.input_dim)
        weights = np.random.uniform(low=min_w, high=max_w, size=[self.output_dim, self.input_dim])
        self.weights = Parameter(weights, transform=positive)
        bias = 0.01*np.ones([self.output_dim], dtype=settings.float_type)
        self.bias = Parameter(bias, transform=positive)

    @params_as_tensors
    def forward(self, input):
        return tf.matmul(input, tf.transpose(self.weights)) + self.bias

    def symbolic(self, ks):
        out = []
        for i in range(self.output_dim):
            tmp = self.bias.value[i]
            w = self.weights.value
            for j in range(self.input_dim):
                tmp = tmp + ks[j] * w[i, j]
            out.append(tmp)
        return out


class Product(_KernelLayer):
    def __init__(self, input_dim, step, name):
        super(Product, self).__init__(input_dim, name)
        assert isinstance(step, int) and step > 1, 'step must be number greater than 1'
        assert int(math.fmod(input_dim, step)) == 0, 'input dim must be multiples of step'
        self.step = step

    @params_as_tensors
    def forward(self, input):
        output = tf.reshape(input, [tf.shape(input)[0], -1, self.step])
        output = tf.reduce_prod(output, -1)
        return output

    def symbolic(self, ks):
        out = []
        for i in range(int(self.input_dim / self.step)):
            out.append(np.prod(ks[i*self.step : (i+1)*self.step]))
        return out


_LAYERS = dict(
    Linear=Linear,
    Product=Product
)


class NKNWrapper(Parameterized):

    def __init__(self, config, name='nkn'):
        Parameterized.__init__(self, name=name)
        self.layers = [_LAYERS[l['name']](**l['params']) for l in config]

    @property
    def params(self):
        for key, param in sorted(self.__dict__.items()):
            if not key.startswith('_') and Parameterized._is_param_like(param):
                yield param
            if not key.startswith('_') and isinstance(param, list):
                for item in param:
                    if Parameterized._is_param_like(item):
                        yield item

    def forward(self, input):
        # input: [n*m, n_primitive_kernels]
        outputs = input
        for l in self.layers:
            outputs = l.forward(outputs)
        # outputs: [n*m, 1]
        return outputs

    def symbolic(self):
        """
        return symbolic formula for the whole network.
        """
        ks = sp.symbols(['k'+str(i) for i in range(self.layers[0].input_dim)]) + [1.]
        for l in self.layers:
            ks = l.symbolic(ks)
        assert len(ks) == 1, 'output of NKN must only have one term'
        return ks[0]


_KERNEL_DICT = dict(
    White=gfk.White,
    Constant=gfk.Constant,
    ExpQuad=gfk.RBF,
    RBF=gfk.RBF,
    Matern12=gfk.Matern12,
    Matern32=gfk.Matern32,
    Matern52=gfk.Matern52,
    Cosine=gfk.Cosine,
    ArcCosine=gfk.ArcCosine,
    Linear=gfk.Linear,
    Periodic=gfk.Periodic,
    RatQuad=gfk.RationalQuadratic,
)


def KernelWrapper(hparams):
    """
    Wrapper for Kernels.
    :param hyparams: list of dict. Each item corresponds to one primitive kernel.
        The dict is formatted as {'name': XXX, 'params': XXX}.
        e.g.
            [{'name': 'Linear', params={'c': 0.1, 'input_dim': 100}},
             {'name': 'Periodic', params={'period': 2, 'input_dim': 100, 'ls': 2}}]
    """
    assert len(hparams) > 0, 'At least one kernel should be provided.'
    return [_KERNEL_DICT[k['name']](**k['params']) for k in hparams]


class NeuralKernelNetwork(Kernel):
    """
    The Neural-Kernel-Network kernel.
    """

    def __init__(self, input_dim, primitive_kernels, nknWrapper, active_dims=None, name='NKN'):
        super(NeuralKernelNetwork, self).__init__(input_dim, active_dims, name=name)

        self.primitive_kernels = primitive_kernels
        self.nknWrapper = nknWrapper

    @property
    def params(self):
        for key, param in sorted(self.__dict__.items()):
            if not key.startswith('_') and Parameterized._is_param_like(param):
                yield param
            if not key.startswith('_') and isinstance(param, list):
                for item in param:
                    if Parameterized._is_param_like(item):
                        yield item

    @params_as_tensors
    def Kdiag(self, X, presliced=False):
        primitive_values = [kern.Kdiag(X) if isinstance(kern, Static) else kern.Kdiag(X, presliced)
                            for kern in self.primitive_kernels]
        primitive_values = tf.stack(primitive_values, 1)
        nkn_outputs = self.nknWrapper.forward(primitive_values)
        return tf.squeeze(nkn_outputs, -1)

    @params_as_tensors
    def K(self, X, X2=None, presliced=False):
        primitive_values = [kern.K(X, X2, presliced) for kern in self.primitive_kernels]
        dynamic_shape_ = tf.shape(primitive_values[0])
        primitive_values = [tf.reshape(val, [-1]) for val in primitive_values]
        primitive_values = tf.stack(primitive_values, 1)
        nkn_outputs = self.nknWrapper.forward(primitive_values)
        return tf.reshape(nkn_outputs, dynamic_shape_)
