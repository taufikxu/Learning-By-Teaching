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
    # import seaborn as sns
    import matplotlib.pyplot as plt
    plt.switch_backend('Agg')

    import utils
    import paramgraphics
    import nn
    from tensorflow.contrib.framework.python.ops import arg_scope
    # import tensorflow.contrib.layers as layers

    # ----------------------------------------------------------------
    # Arguments and Settings
    args.message = 'CLA-mnist_' + args.message
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

    # ----------------------------------------------------------------
    # Dataset
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets('MNIST')

    # data_channel = 3
    z_dim = 256

    # ----------------------------------------------------------------
    # Model setup
    logger.info("Setting up model ...")

    def classifier(x, Reuse=tf.AUTO_REUSE, is_training=True):
        from tensorflow.contrib import layers

        def leaky_relu(x, alpha=0.1):
            return tf.maximum(alpha * x, x)

        filters = [32, 64, 64, 64, 128, 128, 128, 256, 256, 256]
        # filters = [int(num * 1.4) for num in filters]
        norm_prms = {'is_training': is_training, 'decay': 0.99, 'scale': True}
        with tf.variable_scope("classifier", reuse=Reuse), \
            arg_scope([layers.conv2d],
                      normalizer_fn=layers.batch_norm,
                      activation_fn=lambda x: leaky_relu(x, 0.1),
                      normalizer_params=norm_prms):

            x = layers.conv2d(x, filters[0], 5)
            x = layers.dropout(x, 0.5, is_training=is_training)
            x = layers.conv2d(x, filters[1], 3)
            x = layers.conv2d(x, filters[2], 3)
            x = layers.conv2d(x, filters[3], 3)
            x = layers.max_pool2d(x, 2, 2)
            x = layers.dropout(x, 0.5, is_training=is_training)
            x = layers.conv2d(x, filters[4], 3)
            x = layers.conv2d(x, filters[5], 3)
            x = layers.conv2d(x, filters[6], 3)
            x = layers.max_pool2d(x, 2, 2)
            x = layers.dropout(x, 0.5, is_training=is_training)
            x = layers.conv2d(x, filters[7], 3)
            x = layers.conv2d(x, filters[8], 3)
            x = layers.conv2d(x, filters[9], 3)
            x = tf.reduce_mean(x, [1, 2])
            logits = layers.fully_connected(x, 10, activation_fn=None)
            return logits

    def generator(z, Reuse=tf.AUTO_REUSE, flatten=False, is_training=True):
        if args.g_nonlin == 'relu':
            print("Use Relu in G")
            nonlin = tf.nn.relu
        else:
            print("Use tanh in G")
            nonlin = tf.nn.tanh

        # nonlin = tf.nn.relu if args.g_nonlin == 'relu' else tf.nn.tanh
        feature_dim = 16
        # norm_prms = {'is_training': is_training, 'decay': 0.9, 'scale': False}
        with tf.variable_scope("generator", reuse=Reuse):

            # lx = layers.fully_connected(z, 1024)
            lx = tf.layers.dense(z, 1024, use_bias=False)
            lx = tf.layers.batch_normalization(lx, training=is_training)
            lx = nonlin(lx)

            lx = tf.layers.dense(lx, feature_dim * 2 * 7 * 7, use_bias=False)
            lx = tf.layers.batch_normalization(lx, training=is_training)
            lx = nonlin(lx)
            lx = tf.reshape(lx, [-1, 7, 7, feature_dim * 2])

            lx = tf.layers.conv2d_transpose(
                lx, feature_dim, 5, 2, use_bias=False, padding='same')
            lx = tf.layers.batch_normalization(lx, training=is_training)
            lx = nonlin(lx)

            lx = tf.layers.conv2d_transpose(lx, 3, 5, 2, padding='same')
            lx = tf.nn.sigmoid(lx)

            if flatten is True:
                lx = tf.layers.flatten(lx)
            return lx

    # def generator(z, Reuse=tf.AUTO_REUSE, flatten=False, is_training=True):
    #     nonlin = tf.nn.relu

    #     # norm_prms = {'is_training': is_training, 'decay': 0.9, 'scale': False}
    #     with tf.variable_scope("generator", reuse=Reuse):

    #         lx = tf.layers.dense(z, 4 * 4 * 64, use_bias=False)
    #         lx = tf.reshape(lx, [-1, 4, 4, 64])
    #         lx = tf.layers.batch_normalization(lx, training=is_training)
    #         lx = nonlin(lx)

    #         lx = tf.layers.conv2d_transpose(
    #             lx, 32, 3, 2, use_bias=False, padding='same')
    #         lx = lx[:, :7, :7, :]
    #         lx = tf.layers.batch_normalization(lx, training=is_training)
    #         lx = nonlin(lx)

    #         lx = tf.layers.conv2d_transpose(
    #             lx, 16, 3, 2, use_bias=False, padding="same")
    #         lx = tf.layers.batch_normalization(lx, training=is_training)
    #         lx = nonlin(lx)

    #         lx = tf.layers.conv2d_transpose(
    #             lx, 8, 3, 2, use_bias=False, padding="same")
    #         lx = tf.layers.batch_normalization(lx, training=is_training)
    #         lx = nonlin(lx)

    #         lx = tf.layers.conv2d_transpose(lx, 3, 3, 1, padding='same')
    #         lx = tf.nn.sigmoid(lx)

    #         if flatten is True:
    #             lx = tf.layers.flatten(lx)
    #         return lx

    lr_train = tf.placeholder(tf.float32, (), "learning_rate")
    flag_training = tf.placeholder(tf.bool, (), "flag_training")

    samples_dat = tf.placeholder(tf.float32, [None, 28, 28, 1], "data")
    labels_dat = tf.placeholder(tf.int32, [None, 10], "label")
    samples_gen = generator(tf.random_normal((100, z_dim), dtype=tf.float32))

    pred_logit = classifier(samples_dat)
    pred_id = tf.argmax(pred_logit, axis=1)
    pred_loss = tf.nn.softmax_cross_entropy_with_logits_v2(
        logits=pred_logit, labels=labels_dat)

    def precision(predictions, labels, average=True):
        tpreds = tf.argmax(predictions, axis=1)
        tlabels = tf.argmax(labels, axis=1)
        prec = tf.to_float(tf.equal(tpreds, tlabels))
        prec = tf.reduce_mean(prec) if average is True else prec
        return prec

    pred_prec = precision(pred_logit, labels_dat)
    pred_prob = tf.nn.softmax(pred_logit)
    pred_p = tf.reduce_max(pred_prob, -1)

    gen_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "generator")
    cla_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "classifier")
    cla_bn_updateop = tf.get_collection(tf.GraphKeys.UPDATE_OPS, "classifier")

    cla_opt = tf.train.AdamOptimizer(lr_train)
    with tf.control_dependencies(cla_bn_updateop):
        cla_ops = cla_opt.minimize(pred_loss, var_list=cla_vars)

    def np_one_hot(x, depth=None, dtype='int32'):
        if np.ndim(x) == 2:
            return x
        if depth is None:
            depth = np.max(x)
        one_hot = np.zeros([x.shape[0], depth])
        one_hot[np.arange(x.shape[0]), x.astype('int32')] = 1.
        return one_hot.astype(dtype)

    # ----------------------------------------------------------------
    # Training
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    cla_saver = tf.train.Saver(max_to_keep=None, var_list=cla_vars)
    if args.cla_path:
        logger.info("Restore classifier")
        cla_saver.restore(sess, args.cla_path)

    gen_saver = tf.train.Saver(max_to_keep=None, var_list=gen_vars)
    if args.gen_path:
        logger.info("Restore generator")
        gen_saver.restore(sess, args.gen_path)

    # # print variables
    # logger.info("Generator parameters:")
    # for p in gen_vars:
    #     logger.debug("%s: %s" % (p.name, sess.run(tf.shape(p))))
    # logger.info("Estimator parameters:")
    # for p in est_vars:
    #     logger.debug("%s: %s" % (p.name, sess.run(tf.shape(p))))
    # logger.info("Adam parameters:")
    # for p in adam_vars:
    #     logger.debug("%s: %s" % (p.name, sess.run(tf.shape(p))))

    def kldiver(pred_prob):
        ratio = pred_prob * 1000. + 1e-20
        kl = np.sum(pred_prob * np.log(ratio))
        return kl

    def inception_score(pred_probs):
        prob_mean = np.mean(pred_probs, 0, keepdims=True)
        KL = pred_probs * (np.log(pred_probs) - np.log(prob_mean))
        KL = np.mean(np.sum(KL, 1))
        scores = np.exp(KL)
        # print(scores)
        return scores

    def train():
        lr = 3e-4
        data = mnist.train

        pl_list, pc_list = [], []
        for itera in range(500 * 100):
            epoch = itera // 500
            if itera % 500 == 0:
                cla_saver.save(
                    sess, os.path.join(dirname, 'classifier_' + str(epoch)))
                if epoch > 50:
                    lr *= 0.95

            batch_x, batch_y = data.next_batch(100)
            batch_x = batch_x.reshape([-1, 28, 28, 1])
            batch_y = np_one_hot(batch_y, 10)
            pl, pc, _ = sess.run(
                [pred_loss, pred_prec, cla_ops],
                feed_dict={
                    lr_train: lr,
                    samples_dat: batch_x,
                    labels_dat: batch_y,
                    flag_training: True
                })

            pl_list.append(pl)
            pc_list.append(pc)
            if itera % 1000 == 0:
                print("train", np.mean(pl_list[-200:]), np.mean(pc_list[-200:]))
                test()

    def test():
        data = mnist.test

        pl_list, pc_list = [], []
        for _ in range(100):
            batch_x, batch_y = data.next_batch(100)
            batch_x = batch_x.reshape([-1, 28, 28, 1])
            batch_y = np_one_hot(batch_y, 10)

            pl, pc = sess.run(
                [pred_loss, pred_prec],
                feed_dict={
                    lr_train: 0.,
                    samples_dat: batch_x,
                    labels_dat: batch_y,
                    flag_training: False
                })
            pl_list.append(pl)
            pc_list.append(pc)
        print("test ", np.mean(pl_list), np.mean(pc_list))

    def evaluate():
        # test()

        modes = np.zeros([1000, 1])
        modes_freq = np.zeros([1000, 1])
        pred_prob_list1 = []
        pred_prob_list2 = []
        pred_prob_list3 = []
        for _ in range(260):
            np_gen_samples = sess.run(samples_gen)

            pid1, pred_p1 = sess.run(
                [pred_id, pred_prob],
                feed_dict={
                    samples_dat: np_gen_samples[:, :, :, 0:1],
                    flag_training: False
                })
            pred_prob_list1.append(pred_p1)

            pid2, pred_p2 = sess.run(
                [pred_id, pred_prob],
                feed_dict={
                    samples_dat: np_gen_samples[:, :, :, 1:2],
                    flag_training: False
                })
            pred_prob_list2.append(pred_p2)

            pid3, pred_p3 = sess.run(
                [pred_id, pred_prob],
                feed_dict={
                    samples_dat: np_gen_samples[:, :, :, 2:3],
                    flag_training: False
                })
            pred_prob_list3.append(pred_p3)

            for i in range(100):
                num = pid1[i] + 10 * pid2[i] + 100 * pid3[i]
                modes_freq[num] += 1
                if modes[num] == 0:
                    modes[num] = 1

        inception_score(np.concatenate(pred_prob_list1, 0))
        inception_score(np.concatenate(pred_prob_list2, 0))
        inception_score(np.concatenate(pred_prob_list3, 0))

        pred_dis = modes_freq.astype(np.float32) / 26000

        logger.info("KL divergence:{}".format(kldiver(pred_dis)))
        logger.info("modes in total:{}".format(np.sum(modes)))
        np.save(os.path.join(dirname, 'mode.npy'), modes_freq)

        image_list = []
        target = [7, 8, 9]
        while (len(image_list) < 100):
            np_gen_samples = sess.run(samples_gen)

            pid1, pred_p1 = sess.run(
                [pred_id, pred_prob],
                feed_dict={
                    samples_dat: np_gen_samples[:, :, :, 0:1],
                    flag_training: False
                })

            pid2, pred_p2 = sess.run(
                [pred_id, pred_prob],
                feed_dict={
                    samples_dat: np_gen_samples[:, :, :, 1:2],
                    flag_training: False
                })

            pid3, pred_p3 = sess.run(
                [pred_id, pred_prob],
                feed_dict={
                    samples_dat: np_gen_samples[:, :, :, 2:3],
                    flag_training: False
                })

            for i in range(100):
                if pid1[i] == target[0] and pid2[i] == target[1] and pid3[i] == target[2]:
                    image_list.append(np_gen_samples[i, :, :, 2])

        # image_list = [
        #     item.reshape([1, 28, 28,
        #                   3]).transpose([0, 3, 1, 2]).reshape([1, 28 * 28 * 3])
        #     for item in image_list
        # ]
        image_list = [item.reshape([1, 28 * 28]) for item in image_list]
        image_list = np.concatenate(image_list[:100], 0)
        paramgraphics.mat_to_img(
            image_list, (28, 28),
            colorImg=False,
            save_path=os.path.join(dirname, 'sample_gen.png'))

    if args.mode == 'train':
        train()
    elif args.mode == 'test':
        test()
    elif args.mode in ['eval', 'evaluate']:
        evaluate()


# ==========================================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-mode', type=str, default="train", help="train, eval or test")

    parser.add_argument(
        '-gpu', type=int, default=0, help="ID of the gpu device to use")

    parser.add_argument(
        '-message', type=str, default="", help="Messages about the experiment")

    parser.add_argument('-loglv', type=str, default="debug", help="Log level")

    parser.add_argument(
        '-seed',
        type=int,
        default=12345,
        help="Numpy random seed for reproducibility")

    parser.add_argument('-cla_path', type=str, default=None, help="")
    parser.add_argument('-gen_path', type=str, default=None, help="")
    parser.add_argument(
        '-g_nonlin', type=str, default="relu", help="nonlin for generator")

    # ----------------------------------------------------------------
    args = parser.parse_args()
    run_experiment(args)
