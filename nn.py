import tensorflow as tf
# from tensorflow.contrib import layers
# import numpy as np
import weakref
import collections
from tensorflow.contrib.framework.python.ops import add_arg_scope


def get_var_maybe_avg(var_name, ema=None, params=None, **kwargs):
    ''' utility for retrieving polyak averaged params '''
    v = tf.get_variable(var_name, **kwargs)
    if params is not None:
        name = v.name.replace("/", "_").replace(":", "_")
        if name in params:
            v = params[name]
        else:
            params[name] = v
    elif ema is not None:
        v = ema.average(v)
    return v


@add_arg_scope
def dense(x,
          in_num,
          out_num,
          name,
          params=None,
          nonlinearity=None,
          ema=None,
          **kwargs):
    ''' fully connected layer '''
    with tf.variable_scope(name):

        # x = tf.layers.flatten(x)
        V = get_var_maybe_avg(
            'V',
            ema=ema,
            params=params,
            shape=[in_num, out_num],
            dtype=tf.float32,
            # initializer=tf.random_normal_initializer(0, 0.05),
            trainable=True)
        # g = get_var_maybe_avg('g', ema, shape=[num_units], dtype=tf.float32,
        #                       initializer=tf.constant_initializer(1.), trainable=True)
        b = get_var_maybe_avg(
            'b',
            ema,
            params=params,
            shape=[out_num],
            dtype=tf.float32,
            initializer=tf.constant_initializer(0.),
            trainable=True)

        # do not use weight normalization (Salimans & Kingma, 2016)
        x = tf.matmul(x, V)
        x = tf.nn.bias_add(x, b)
        # scaler = g / tf.sqrt(tf.reduce_sum(tf.square(V), [0]))
        # x = tf.reshape(scaler, [1, num_units]) * x + \
        #     tf.reshape(b, [1, num_units])

        # apply nonlinearity
        if nonlinearity is not None:
            x = nonlinearity(x)

        return x


@add_arg_scope
def conv2d(x,
           in_num,
           out_num,
           name,
           filter_size=[3, 3],
           params=None,
           stride=[1, 1],
           pad='SAME',
           nonlinearity=None,
           ema=None,
           **kwargs):
    ''' convolutional layer '''
    with tf.variable_scope(name):
        V = get_var_maybe_avg(
            'V',
            ema=ema,
            params=params,
            shape=filter_size + [in_num, out_num],
            dtype=tf.float32,
            initializer=tf.random_normal_initializer(0, 0.05),
            trainable=True)
        # g = get_var_maybe_avg(
        #     'g', ema=ema, params=params,
        #     shape=[num_filters], dtype=tf.float32,
        #     initializer=tf.constant_initializer(1.), trainable=True)
        b = get_var_maybe_avg(
            'b',
            ema=ema,
            params=params,
            shape=[out_num],
            dtype=tf.float32,
            initializer=tf.constant_initializer(0.),
            trainable=True)

        # do not use weight normalization (Salimans & Kingma, 2016)
        # W = tf.reshape(g, [1, 1, 1, num_filters]) * \
        #     tf.nn.l2_normalize(V, [0, 1, 2])

        # calculate convolutional layer output
        x = tf.nn.conv2d(x, V, [1] + stride + [1], pad)
        x = tf.nn.bias_add(x, b)

        # if init:  # normalize x
        #     m_init, v_init = tf.nn.moments(x, [0, 1, 2])
        #     scale_init = init_scale / tf.sqrt(v_init + 1e-10)
        #     with tf.control_dependencies(
        #             [g.assign(g * scale_init), b.assign_add(-m_init * scale_init)]):
        #         x = tf.identity(x)

        # apply nonlinearity
        if nonlinearity is not None:
            x = nonlinearity(x)

        return x


@add_arg_scope
def deconv2d(x,
             in_num,
             out_num,
             target_shape,
             name,
             ema=None,
             params=None,
             filter_size=[3, 3],
             stride=[1, 1],
             pad='SAME',
             nonlinearity=None,
             **kwargs):
    ''' transposed convolutional layer '''

    with tf.variable_scope(name):
        V = get_var_maybe_avg(
            'V',
            ema=ema,
            params=params,
            shape=filter_size + [out_num, in_num],
            dtype=tf.float32,
            initializer=tf.random_normal_initializer(0, 0.05),
            trainable=True)
        # g = get_var_maybe_avg(
        #     'g', ema=ema, params=params,
        #     shape=[num_filters], dtype=tf.float32,
        #     initializer=tf.constant_initializer(1.), trainable=True)
        b = get_var_maybe_avg(
            'b',
            ema=ema,
            params=params,
            shape=[out_num],
            dtype=tf.float32,
            initializer=tf.constant_initializer(0.),
            trainable=True)

        # use weight normalization (Salimans & Kingma, 2016)
        # W = tf.reshape(g, [1, 1, num_filters, 1]) * \
        #     tf.nn.l2_normalize(V, [0, 1, 3])

        # calculate convolutional layer output
        x = tf.nn.conv2d_transpose(
            x, V, target_shape, [1] + stride + [1], padding=pad)
        x = tf.nn.bias_add(x, b)

        # if init:  # normalize x
        #     m_init, v_init = tf.nn.moments(x, [0, 1, 2])
        #     scale_init = init_scale / tf.sqrt(v_init + 1e-10)
        #     with tf.control_dependencies(
        #             [g.assign(g * scale_init), b.assign_add(-m_init * scale_init)]):
        #         x = tf.identity(x)

        # apply nonlinearity
        if nonlinearity is not None:
            x = nonlinearity(x)

        return x


def bn(x):
    from tensorflow.contrib import layers
    return layers.batch_norm(x)


@add_arg_scope
def batch_norm(x,
               name,
               out_num,
               data_rank=4,
               decay=0.99,
               training=True,
               center=True,
               scale=True,
               nonlinearity=None,
               **kwargs):
    ''' transposed convolutional layer '''
    with tf.variable_scope(name):
        if data_rank == 2:
            params_shape = [1, out_num]
            mean_axis = [0]
        else:
            params_shape = [1, 1, 1, out_num]
            mean_axis = [0, 1, 2]

        if center is True:
            beta = tf.get_variable(
                "beta",
                shape=params_shape,
                dtype=tf.float32,
                initializer=tf.constant_initializer(0.))
        else:
            beta = 0.

        if scale is True:
            gamma = tf.get_variable(
                "scale",
                shape=params_shape,
                dtype=tf.float32,
                initializer=tf.constant_initializer(1.))
        else:
            gamma = 1.

        moving_mean = tf.get_variable(
            "moving_mean",
            shape=params_shape,
            dtype=tf.float32,
            initializer=tf.constant_initializer(0.))
        moving_var = tf.get_variable(
            "moving_var",
            shape=params_shape,
            dtype=tf.float32,
            initializer=tf.constant_initializer(1.))

        if training is True:
            curmean, curvar = tf.nn.moments(x, axes=mean_axis, keep_dims=True)
            normalized_x = (x - curmean) / (curvar + 1e-6)
            output = normalized_x * gamma + beta

            update_moving_avg = tf.assign(
                moving_mean, decay * moving_mean + (1 - decay) * curmean)
            update_moving_var = tf.assign(
                moving_var, decay * moving_var + (1 - decay) * curmean)
            updates_collections = tf.GraphKeys.UPDATE_OPS
            tf.add_to_collections(updates_collections, update_moving_avg)
            tf.add_to_collections(updates_collections, update_moving_var)
        else:
            normalized_x = (x - moving_mean) / (moving_var + 1e-6)
            output = normalized_x * gamma + beta

        if nonlinearity is not None:
            output = nonlinearity(output)

        return output


def kernel_density_estimator(data, sample, std=0.05):
    data = tf.layers.flatten(data)
    sample = tf.layers.flatten(sample)

    data = tf.expand_dims(data, 1)
    sample = tf.expand_dims(sample, 0)

    kmatrix = tf.reduce_sum(-1 * tf.square(sample - data), -1)
    kmatrix = kmatrix / (std**2)

    return tf.reduce_logsumexp(kmatrix, -1, keepdims=False)
