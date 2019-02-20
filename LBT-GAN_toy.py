from __future__ import division

# ==========================================================================


def run_experiment(args):
    import os
    # set environment variables for tensorflow
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    import inspect
    import shutil
    import numpy as np
    import tensorflow as tf

    from collections import OrderedDict
    import matplotlib.pyplot as plt
    plt.switch_backend('Agg')
    import seaborn as sns

    import utils
    from Config.toy_data import ring_mog, grid_mog
    import nn
    from tensorflow.contrib.framework.python.ops import arg_scope


# ----------------------------------------------------------------
# Arguments and Settings
    args.message = 'LBT-GAN-toy_' + args.message
    np.random.seed(args.seed)
    tf.set_random_seed(args.seed)

    # copy file for reproducibility
    logger, dirname = utils.setup_logging(args)
    script_fn = inspect.getfile(inspect.currentframe())
    script_src = os.path.abspath(script_fn)
    script_dst = os.path.abspath(os.path.join(dirname, script_fn))
    shutil.copyfile(script_src, script_dst)
    logger.info("script copied from %s to %s" % (script_src, script_dst))

    # print arguments
    for k, v in sorted(vars(args).items()):
        logger.info("  %20s: %s" % (k, v))

    # get arguments
    batch_size = args.batch_size
    batch_size_est = args.batch_size_est
    gen_lr = args.gen_lr
    dis_lr = args.dis_lr
    est_lr = args.est_lr
    lambda_gan = args.lambda_gan
    beta1 = 0.5
    epsilon = 1e-8
    max_iter = args.max_iter
    viz_every = args.viz_every
    z_dim, vae_z_dim = utils.get_ints(args.z_dims)
    n_mixture = args.n_mixture
    mog_std = args.mog_std
    mog_scale = args.mog_scale
    vae_x_var = 0.001
    unrolling_steps = args.unrolling_steps
    assert unrolling_steps > 0
    n_viz = args.n_viz

# ----------------------------------------------------------------
# Dataset
    x_dim = 2

    if args.dataset == 'ring':
        data, modes = ring_mog(batch_size, std=mog_std, radius=mog_scale)
    elif args.dataset == 'grid':
        data, modes = grid_mog(
            batch_size, n_mixture=n_mixture, std=mog_std, space=mog_scale)
    else:
        raise ValueError("Unsupported Data Type: %s" % args.dataset)

# ----------------------------------------------------------------
# Model setup
    logger.info("Setting up model ...")

    def discriminator(x, n_hidden=128, reuse=tf.AUTO_REUSE, is_training=True):
        with tf.variable_scope("discriminator", reuse=reuse):
            h = tf.layers.dense(inputs=x, units=n_hidden,
                                activation=tf.nn.tanh)
            h = tf.layers.dense(inputs=h, units=n_hidden,
                                activation=tf.nn.tanh)
            log_d = tf.layers.dense(inputs=h, units=1,
                                    activation=None)
        return log_d

    def generator(z, n_hidden=128, reuse=tf.AUTO_REUSE):
        with tf.variable_scope("generator", reuse=reuse):
            h = tf.layers.dense(inputs=z, units=n_hidden,
                                activation=tf.nn.tanh)
            h = tf.layers.dense(inputs=h, units=n_hidden,
                                activation=tf.nn.tanh)
            x = tf.layers.dense(inputs=h, units=x_dim, activation=None)
        return x

    def nonlin(x_):
        return tf.nn.tanh(x_)

    def compute_est_samples(z, params=None, reuse=tf.AUTO_REUSE):
        with tf.variable_scope("estimator"):
            with arg_scope([nn.dense], params=params):
                with tf.variable_scope("decoder", reuse=reuse):
                    h_dec_1 = nn.dense(z, vae_z_dim, 128, "dense1", nonlinearity=nonlin)
                    h_dec_2 = nn.dense(h_dec_1, 128, 128, "dense2", nonlinearity=nonlin)
                    x_mean = nn.dense(h_dec_2, 128, x_dim, "dense3", nonlinearity=None)
                    return x_mean

    def compute_est_ll(x, params=None, reuse=tf.AUTO_REUSE):
        with tf.variable_scope("estimator"):
            with arg_scope([nn.dense], params=params):
                with tf.variable_scope("encoder", reuse=reuse):
                    h_enc_1 = nn.dense(x, 2, 128, "dense1", nonlinearity=nonlin)
                    # h_enc_1 = nn.batch_norm(h_enc_1, "bn1", 128, 2)
                    h_enc_2 = nn.dense(h_enc_1, 128, 128, "dense2", nonlinearity=nonlin)
                    # h_enc_2 = nn.batch_norm(h_enc_2, "bn2", 128, 2)
                    z_mean = nn.dense(h_enc_2, 128, vae_z_dim, "dense3", nonlinearity=None)
                    z_logvar = nn.dense(h_enc_2, 128, vae_z_dim, "dense4", nonlinearity=None)
                epsilon = tf.random_normal(tf.shape(z_mean), dtype=tf.float32)
                z = z_mean + tf.exp(0.5 * z_logvar) * epsilon

                with tf.variable_scope("decoder", reuse=reuse):
                    h_dec_1 = nn.dense(z, vae_z_dim, 128, "dense1", nonlinearity=nonlin)
                    # h_dec_1 = nn.batch_norm(h_dec_1, "bn1", 128, 2)
                    h_dec_2 = nn.dense(h_dec_1, 128, 128, "dense2", nonlinearity=nonlin)
                    # h_dec_2 = nn.batch_norm(h_dec_2, "bn2", 128, 2)
                    x_mean = nn.dense(h_dec_2, 128, x_dim, "dense3", nonlinearity=None)

        elbo = tf.reduce_mean(tf.reduce_sum(- 0.5 * np.log(2 * np.pi) - 0.5 * np.log(vae_x_var) - tf.square(x - x_mean) / (
            2 * vae_x_var), axis=1) - tf.reduce_sum(- 0.5 * (1 + z_logvar - tf.square(z_mean) - tf.exp(z_logvar)), axis=1))
        return elbo, x_mean

    def compute_est_updated_with_SGD(x, lr=0.001, params=None):
        elbo, _ = compute_est_ll(x, params=params)
        grads = tf.gradients(elbo, params.values())
        new_params = params.copy()
        for key, g in zip(params, grads):
            new_params[key] += lr * g
        return elbo, new_params

    def compute_est_updated_with_Adam(x, lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7, decay=0., params=None, adam_params=None):
        elbo, _ = compute_est_ll(x, params=params)
        grads = tf.gradients(elbo, params.values())
        new_params = params.copy()
        new_adam_params = adam_params.copy()
        new_adam_params['iterations'] += 1
        lr = lr * \
            (1. / (1. + decay *
                   tf.cast(adam_params['iterations'], tf.float32)))
        t = tf.cast(new_adam_params['iterations'], tf.float32)
        lr_t = lr * (tf.sqrt(1. - tf.pow(beta_2, t)) /
                     (1. - tf.pow(beta_1, t)))
        for key, g in zip(params, grads):
            new_adam_params['m_' + key] = (beta_1 * adam_params['m_' + key]) + (1. - beta_1) * g
            new_adam_params['v_' + key] = tf.stop_gradient((beta_2 * adam_params['v_' + key]) + (1. - beta_2) * tf.square(g))
            new_params[key] = params[key] + lr_t * new_adam_params['m_' + key] / tf.sqrt(new_adam_params['v_' + key] + epsilon)
        return elbo, new_params, new_adam_params

    # tf.reset_default_graph()
    lr = tf.placeholder(tf.float32)

    # Construct generator and estimator nets
    est_params_dict = OrderedDict()
    gen_noise = tf.random_normal((batch_size_est, z_dim), dtype=tf.float32)
    samples = generator(gen_noise)
    sample_elbo, _ = compute_est_ll(samples, params=est_params_dict)
    vae_noise = tf.random_normal((batch_size_est, vae_z_dim), dtype=tf.float32)
    samples_est = compute_est_samples(z=vae_noise, params=est_params_dict)
    # for key in est_params_dict:
    #     print(key, est_params_dict[key])

    adam_params_dict = OrderedDict()
    with tf.variable_scope("adam"):
        adam_params_dict['iterations'] = tf.Variable(
            0, dtype=tf.int64, name='iterations')
        for key in est_params_dict:
            adam_params_dict['m_' + key] = tf.Variable(
                tf.zeros_like(est_params_dict[key]), name='m_' + key)
            adam_params_dict['v_' + key] = tf.Variable(
                tf.zeros_like(est_params_dict[key]), name='v_' + key)

    gen_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "generator")
    est_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "estimator")
    adam_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "adam")

    # unrolling estimator updates

    cur_params = est_params_dict
    cur_adam_params = adam_params_dict
    elbo_genx_at_steps = []
    for unroll in range(unrolling_steps):
        samples = generator(
            tf.random_normal((batch_size_est, z_dim), dtype=tf.float32))
        elbo_genx_step, cur_params, cur_adam_params = compute_est_updated_with_Adam(
            samples, lr=lr, beta_1=beta1, epsilon=epsilon, params=cur_params, adam_params=cur_adam_params)
        # for key in cur_params:
        #     print(key, cur_params[key])
        # elbo_genx_step, cur_params = compute_est_updated_with_SGD(samples, lr=lr, params=cur_params)
        elbo_genx_at_steps.append(elbo_genx_step)

    # estimator update
    updates = []
    for key in est_params_dict:
        updates.append(tf.assign(est_params_dict[key], cur_params[key]))
    for key in adam_params_dict:
        updates.append(tf.assign(adam_params_dict[key], cur_adam_params[key]))
    e_train_op = tf.group(*updates, name="e_train_op")

    # Optimize the generator on the unrolled ELBO loss
    unrolled_elbo_data, _ = compute_est_ll(data, params=cur_params)

    # GAN-loss for discriminator and generator
    samples_gen = generator(
        tf.random_normal((batch_size_est, z_dim), dtype=tf.float32))
    fake_D_output = discriminator(samples_gen)
    real_D_output = discriminator(data)
    ganloss_g = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.ones_like(fake_D_output), logits=fake_D_output))
    ganloss_D_fake = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.zeros_like(fake_D_output), logits=fake_D_output))
    ganloss_D_real = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.ones_like(real_D_output), logits=real_D_output))

    object_g = lambda_gan * ganloss_g - unrolled_elbo_data
    # object_g = -1 * unrolled_elbo_data
    object_d = ganloss_D_fake + ganloss_D_real
    dis_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                 "discriminator")

    g_train_opt = tf.train.AdamOptimizer(
        learning_rate=gen_lr, beta1=beta1, epsilon=epsilon)
    # g_train_opt = tf.train.RMSPropOptimizer(learning_rate=gen_lr, epsilon=epsilon)
    g_grads = g_train_opt.compute_gradients(object_g, var_list=gen_vars)
    # g_grads_clipped = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in g_grads]
    g_grads_, g_vars_ = zip(*g_grads)
    g_grads_clipped_, g_grads_norm_ = tf.clip_by_global_norm(g_grads_, 5.)
    g_grads_clipped = zip(g_grads_clipped_, g_vars_)
    if not args.no_clip_grad:
        logger.info("Clipping gradients of generator parameters.")
        g_train_op = g_train_opt.apply_gradients(g_grads_clipped)
    else:
        g_train_op = g_train_opt.apply_gradients(g_grads)

    d_train_opt = tf.train.AdamOptimizer(
        learning_rate=dis_lr, beta1=beta1, epsilon=epsilon)
    d_train_op = d_train_opt.minimize(object_d, var_list=dis_vars)

# ----------------------------------------------------------------
# Training
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(max_to_keep=None)
    if args.model_path:
        saver.restore(sess, args.model_path)

    # print variables
    logger.info("Generator parameters:")
    for p in gen_vars:
        logger.debug("%s: %s" % (p.name, sess.run(tf.shape(p))))
    logger.info("Estimator parameters:")
    for p in est_vars:
        logger.debug("%s: %s" % (p.name, sess.run(tf.shape(p))))
    logger.info("Adam parameters:")
    for p in adam_vars:
        logger.debug("%s: %s" % (p.name, sess.run(tf.shape(p))))

    elbo_vals = []
    ganloss_vals = []
    tgan_g, tgan_d_fake, tgan_d_real = 0., 0., 0.
    elbo_genx_val, elbo_data_val, gradients_nrom = -np.inf, -np.inf, 0
    for i in xrange(max_iter + 1):
        # visualization
        if i % viz_every == 0:
            np_samples_data = np.vstack(
                [sess.run(data) for _ in xrange(n_viz // batch_size + 1)])[:n_viz]
            np_samples_gen = np.vstack(
                [sess.run(samples) for _ in xrange(n_viz // batch_size_est + 1)])[:n_viz]
            np_samples_est = np.vstack(
                [sess.run(samples_est) for _ in xrange(n_viz // batch_size_est + 1)])[:n_viz]
            fig = plt.figure(figsize=(5, 5))
            plt.scatter(np_samples_data[:, 0], np_samples_data[:, 1],
                        s=8, c='r', edgecolor='none', alpha=0.05)
            plt.scatter(np_samples_est[:, 0], np_samples_est[:, 1],
                        s=8, c='g', edgecolor='none', alpha=0.05)
            plt.scatter(np_samples_gen[:, 0], np_samples_gen[:, 1],
                        s=8, c='b', edgecolor='none', alpha=0.05)
            if args.dataset == 'ring':
                plt.xlim((-1.5 * mog_scale, 1.5 * mog_scale))
                plt.ylim((-1.5 * mog_scale, 1.5 * mog_scale))
            if args.dataset == 'grid':
                plt.xlim((modes[0][0] - 1.5 * mog_scale,
                          modes[-1][-1] + 1.5 * mog_scale))
                plt.ylim((modes[0][0] - 1.5 * mog_scale,
                          modes[-1][-1] + 1.5 * mog_scale))
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(dirname, 'sample_' +
                                     str(i) + '.png'), bbox_inches='tight')
            plt.close(fig)

            fig = plt.figure(figsize=(6, 4))
            plt.plot(elbo_vals, '.', markersize=2, markeredgecolor='none',
                     linestyle='none', alpha=min(1.0, 0.01 * max_iter / (i + 1)))
            plt.ylim((-4.0, 0.5))
            legend = plt.legend(('elbo_genx', 'elbo_data'), markerscale=6)
            for lh in legend.legendHandles:
                lh._legmarker.set_alpha(1.)
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(dirname, 'curve.png'),
                        bbox_inches='tight')
            plt.close(fig)

        # train estimator and generator
        for _ in xrange(args.n_est):
            elbo_genx_val, _ = sess.run(
                [elbo_genx_at_steps[-1], e_train_op], feed_dict={lr: 3. * est_lr})

        for _ in range(args.n_dis):
            _, tgan_g, tgan_d_real, tgan_d_fake = sess.run(
                [d_train_op, ganloss_g, ganloss_D_real, ganloss_D_fake])

        elbo_data_val, gradients_nrom, _ = sess.run(
            [unrolled_elbo_data, g_grads_norm_, g_train_op], feed_dict={lr: est_lr})
        elbo_vals.append([elbo_genx_val, elbo_data_val])
        ganloss_vals.append([tgan_g, tgan_d_real, tgan_d_fake])

        # training log
        if i % viz_every == 0:
            elbo_genx_ma_val, elbo_data_ma_val = np.mean(
                elbo_vals[-200:], axis=0)
            logger.info("Iter %d: gradients norm = %.4f. samples LL = %.4f, data LL = %.4f." % (
                i, gradients_nrom, elbo_genx_ma_val, elbo_data_ma_val))
            logger.info(
                "Iter %d: gan_g = %.4f. gan_d_real = %.4f, gan_d_fake = %.4f." %
                (i, tgan_g, tgan_d_real, tgan_d_fake))

        if i % args.model_every == 0:
            saver.save(sess, os.path.join(dirname, 'model_' + str(i)))


# ==========================================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('-gpu', type=int, default=0,
                        help="ID of the gpu device to use")

    parser.add_argument('-message', type=str, default="",
                        help="Messages about the experiment")

    parser.add_argument('-loglv', type=str, default="debug",
                        help="Log level")

    parser.add_argument('-dataset', type=str, default="ring",
                        help="Dataset name")

    parser.add_argument('-seed', type=int, default=12345,
                        help="Numpy random seed for reproducibility")

    parser.add_argument('-model_path', type=str, default=None,
                        help="")

# ----------------------------------------------------------------

    parser.add_argument('-z_dims', type=str, default='256,32',
                        help="Mini-batch size")

    parser.add_argument('-batch_size', type=int, default=512,
                        help="Mini-batch size")

    parser.add_argument('-batch_size_est', type=int, default=512,
                        help="Mini-batch size")

    parser.add_argument('-gen_lr', type=float, default=1e-3,
                        help="")

    parser.add_argument('-dis_lr', type=float, default=1e-4,
                        help="")

    parser.add_argument('-est_lr', type=float, default=1e-4,
                        help="")

    parser.add_argument('-lambda_gan', type=float, default=0.1, help="")

    parser.add_argument('-n_est', type=int, default=2,
                        help="")

    parser.add_argument('-n_dis', type=int, default=1, help="")

    parser.add_argument('-no_clip_grad', default=False, action="store_true",
                        help="")

    parser.add_argument('-n_mixture', type=int, default=25,
                        help="# mixtures of Grid dataset")

    parser.add_argument('-mog_std', type=float, default=0.05,
                        help="")

    parser.add_argument('-mog_scale', type=float, default=1.,
                        help="")

    parser.add_argument('-max_iter', type=int, default=1000000,
                        help="Maximum iterations")

    parser.add_argument('-viz_every', type=int, default=10000,
                        help="")

    parser.add_argument('-n_viz', type=int, default=5120,
                        help="")

    parser.add_argument('-model_every', type=int, default=100000,
                        help="")

    parser.add_argument('-unrolling_steps', type=int, default=5,
                        help="")

# ----------------------------------------------------------------
    args = parser.parse_args()
    run_experiment(args)
