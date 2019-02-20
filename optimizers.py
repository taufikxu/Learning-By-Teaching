"""Adam implementation based on keras.optimizers
"""
from __future__ import division
import tensorflow as tf


class Optimizer(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'clipnorm', 'clipvalue'}
        for k in kwargs:
            if k not in allowed_kwargs:
                raise TypeError('Unexpected keyword argument '
                                'passed to optimizer: ' + str(k))
        self.__dict__.update(kwargs)
        self.updates = []
        self.weights = []

    def get_updates(self, loss, params):
        raise NotImplementedError

    def get_gradients(self, loss, params):
        grads = tf.gradients(loss, params, colocate_gradients_with_ops=True)
        if None in grads:
            raise ValueError('An operation has `None` for gradient. '
                             'Please make sure that all of your ops have a '
                             'gradient defined (i.e. are differentiable). ')
        if hasattr(self, 'clipnorm') and self.clipnorm > 0:
            grads, grads_norm = tf.clip_by_global_norm(grads, self.clipnorm)
        if hasattr(self, 'clipvalue') and self.clipvalue > 0:
            grads = [tf.clip_by_value(g, -self.clipvalue, self.clipvalue) for g in grads]
        return grads


class Adam(Optimizer):
    """Adam optimizer.

    Default parameters follow those provided in the original paper.

    # Arguments
        lr: float >= 0. Learning rate.
        beta_1: float, 0 < beta < 1. Generally close to 1.
        beta_2: float, 0 < beta < 1. Generally close to 1.
        epsilon: float >= 0. Fuzz factor. Defaults to 1e-7.
        decay: float >= 0. Learning rate decay over each update.
        amsgrad: boolean. Whether to apply the AMSGrad variant of this
            algorithm from the paper "On the Convergence of Adam and
            Beyond".

    # References
        - [Adam - A Method for Stochastic Optimization](http://arxiv.org/abs/1412.6980v8)
        - [On the Convergence of Adam and Beyond](https://openreview.net/forum?id=ryQu7f-RZ)
    """

    def __init__(self, lr=0.001, beta_1=0.9, beta_2=0.999,
                 epsilon=1e-7, decay=0., amsgrad=False, **kwargs):
        super(Adam, self).__init__(**kwargs)
        with tf.name_scope(self.__class__.__name__):
            self.iterations = tf.Variable(0, dtype=tf.int64, name='iterations')
            self.lr = tf.Variable(lr, dtype=tf.float32, name='lr')
            self.beta_1 = tf.Variable(beta_1, dtype=tf.float32, name='beta_1')
            self.beta_2 = tf.Variable(beta_2, dtype=tf.float32, name='beta_2')
            self.decay = tf.Variable(decay, dtype=tf.float32, name='decay')
        self.epsilon = epsilon
        self.initial_decay = decay
        self.amsgrad = amsgrad

    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [tf.assign_add(self.iterations, 1)]

        lr = self.lr
        if self.initial_decay > 0:
            lr *= (1. / (1. + self.decay * tf.cast(self.iterations, tf.float32)))

        t = tf.cast(self.iterations, tf.float32) + 1
        lr_t = lr * (tf.sqrt(1. - tf.pow(self.beta_2, t)) / (1. - tf.pow(self.beta_1, t)))

        ms = [tf.Variable(tf.zeros_like(p)) for p in params]
        vs = [tf.Variable(tf.zeros_like(p)) for p in params]
        if self.amsgrad:
            vhats = [tf.Variable(tf.zeros_like(p)) for p in params]
        else:
            vhats = [tf.Variable(tf.zeros(1)) for _ in params]
        self.weights = [self.iterations] + ms + vs + vhats

        for p, g, m, v, vhat in zip(params, grads, ms, vs, vhats):
            m_t = (self.beta_1 * m) + (1. - self.beta_1) * g
            v_t = (self.beta_2 * v) + (1. - self.beta_2) * tf.square(g)
            if self.amsgrad:
                vhat_t = tf.maximum(vhat, v_t)
                # p_t = p - lr_t * m_t / (tf.sqrt(vhat_t) + self.epsilon)
                p_t = p - lr_t * m_t / tf.sqrt(vhat_t + self.epsilon)
                self.updates.append(tf.assign(vhat, vhat_t))
            else:
                # p_t = p - lr_t * m_t / (tf.sqrt(v_t) + self.epsilon)
                p_t = p - lr_t * m_t / tf.sqrt(v_t + self.epsilon)

            self.updates.append(tf.assign(m, m_t))
            self.updates.append(tf.assign(v, v_t))
            new_p = p_t

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(tf.assign(p, new_p))
        return self.updates
