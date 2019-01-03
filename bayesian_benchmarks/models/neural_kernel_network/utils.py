import numpy as np


def median_distance_local(x):
    """
    get the median of distances between x.
    :param x: shape of [n, d]
    :return: float
    """
    if x.shape[0] > 10000:
        permutation = np.random.permutation(x.shape[0])
        x = x[permutation[:10000]]
    x_col = np.expand_dims(x, 1)
    x_row = np.expand_dims(x, 0)
    dis_a = np.abs(x_col - x_row) # [n, n, d]
    dis_a = np.reshape(dis_a, [-1, dis_a.shape[-1]])
    ls = np.median(dis_a, 0) * (x.shape[1] ** 0.5)
    ls[abs(ls) < 1e-6] = 1.
    return ls


def default(input_dim, ls):
    return dict(
        nkn=[
            {'name': 'Linear',  'params': {'input_dim': 6, 'output_dim': 8, 'name': 'layer1'}},
            {'name': 'Product', 'params': {'input_dim': 8, 'step': 2,       'name': 'layer2'}},
            {'name': 'Linear',  'params': {'input_dim': 4, 'output_dim': 4, 'name': 'layer3'}},
            {'name': 'Product', 'params': {'input_dim': 4, 'step': 2,       'name': 'layer4'}},
            {'name': 'Linear',  'params': {'input_dim': 2, 'output_dim': 1, 'name': 'layer5'}}],
        kern=[
            {'name': 'Linear',  'params': {'input_dim': input_dim, 'ARD': True, 'name': 'Linear1'}},
            {'name': 'Linear',  'params': {'input_dim': input_dim, 'ARD': True, 'name': 'Linear2'}},
            {'name': 'RBF',     'params': {'input_dim': input_dim, 'lengthscales': ls / 6., 'ARD': True, 'name': 'RBF1'}},
            {'name': 'RBF',     'params': {'input_dim': input_dim, 'lengthscales': ls / 3. * 2., 'ARD': True, 'name': 'RBF2'}},
            {'name': 'RatQuad', 'params': {'input_dim': input_dim, 'alpha': 0.1, 'lengthscales': ls / 3., 'name': 'RatQuad1'}},
            {'name': 'RatQuad', 'params': {'input_dim': input_dim, 'alpha': 1.0, 'lengthscales': ls / 3., 'name': 'RatQuad2'}}],
    )


def get_nkn_config(name='default'):
    if name == 'default':
        return default
    else:
        raise NameError('Name %s not supported' % name)