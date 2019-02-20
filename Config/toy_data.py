import numpy as np
import itertools
import tensorflow as tf
import tensorflow.contrib.distributions as ds


def ring_mog(batch_size, n_mixture=8, std=0.01, radius=1.0):
    thetas = np.linspace(0, 2 * np.pi, n_mixture, endpoint=False)
    xs = radius * np.sin(thetas, dtype=np.float32)
    ys = radius * np.cos(thetas, dtype=np.float32)
    cat = ds.Categorical(tf.zeros(n_mixture))
    comps = [ds.MultivariateNormalDiag([xi, yi], [std, std]) for xi, yi in
             zip(xs.ravel(), ys.ravel())]
    data = ds.Mixture(cat, comps)
    return data.sample(batch_size), np.stack([xs, ys], axis=1)


def grid_mog(batch_size, n_mixture=25, std=0.05, space=2.0):
    grid_range = int(np.sqrt(n_mixture))
    modes = np.array([np.array([i, j]) for i, j in
                      itertools.product(range(-grid_range + 1, grid_range, 2),
                                        range(-grid_range + 1, grid_range, 2))],
                     dtype=np.float32)
    modes = modes * space / 2.
    cat = ds.Categorical(tf.zeros(n_mixture))
    comps = [ds.MultivariateNormalDiag(mu, [std, std]) for mu in modes]
    data = ds.Mixture(cat, comps)
    return data.sample(batch_size), modes
