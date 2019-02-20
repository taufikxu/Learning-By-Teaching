import sys
import os
import gzip
import numpy
import numpy as np
import pickle
import scipy.io as sio


def to_one_hot(x, depth):
    ret = np.zeros((x.shape[0], depth), dtype=np.int32)
    ret[np.arange(x.shape[0]), x] = 1
    return ret


def load_mnist_realval(path='/home/Data/mnist.pkl.gz',
                       asimage=True,
                       one_hot=False,
                       validation=True,
                       isTf=True):
    """
    return_all flag will return all of the data. It will overwrite validation
    nlabeled.
    """
    if not os.path.isfile(path):
        data_dir = os.path.dirname(path)
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(data_dir)

        def download_dataset(url, _path):
            print('Downloading data from %s' % url)
            if sys.version_info > (2,):
                import urllib.request as request
            else:
                from urllib2 import Request as request
            request.urlretrieve(url, _path)

        download_dataset(
            'http://www.iro.umontreal.ca/~lisa/deep/data/mnist'
            '/mnist.pkl.gz', path)

    with gzip.open(path, 'rb') as f:
        if sys.version_info > (3,):
            train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
        else:
            train_set, valid_set, test_set = pickle.load(f)
    x_train, y_train = train_set[0], train_set[1].astype('int32')
    x_valid, y_valid = valid_set[0], valid_set[1].astype('int32')
    x_test, y_test = test_set[0], test_set[1].astype('int32')

    n_y = y_train.max() + 1
    t_transform = (lambda x: to_one_hot(x, n_y)) if one_hot else (lambda x: x)
    y_train, y_valid = t_transform(y_train), t_transform(y_valid)
    y_test = t_transform(y_test)

    if asimage is True:
        x_train = x_train.reshape([-1, 28, 28, 1])
        x_valid = x_valid.reshape([-1, 28, 28, 1])
        x_test = x_test.reshape([-1, 28, 28, 1])
    if isTf is False:
        x_train = x_train.transpose([0, 3, 1, 2])
        x_valid = x_valid.transpose([0, 3, 1, 2])
        x_test = x_test.transpose([0, 3, 1, 2])

    if validation is True:
        x_test = x_valid
        y_test = y_valid
    else:
        x_train = np.concatenate((x_train, x_valid))
        y_train = np.concatenate((y_train, y_valid))

    return x_train, y_train, x_test, y_test


def load_cifar10(data_dir='/home/Data/cifar/', one_hot=False, isTf=True):

    def file_name(ind):
        return os.path.join(data_dir,
                            'cifar-10-batches-py/data_batch_' + str(ind))

    def unpickle_cifar_batch(file_):
        fo = open(file_, 'rb')
        if sys.version_info > (3,):
            tmp_data = pickle.load(fo, encoding='latin1')
        else:
            tmp_data = pickle.load(fo)
        fo.close()
        x_ = tmp_data['data'].astype(np.float32)
        x_ = x_.reshape((10000, 3, 32, 32)) / 255.
        y_ = np.array(tmp_data['labels']).astype(np.float32)
        return {'x': x_, 'y': y_}

    train_data = [unpickle_cifar_batch(file_name(i)) for i in range(1, 6)]
    x_train = np.concatenate([td['x'] for td in train_data])
    y_train = np.concatenate([td['y'] for td in train_data])
    y_train = y_train.astype('int32')

    test_data = unpickle_cifar_batch(
        os.path.join(data_dir, 'cifar-10-batches-py/test_batch'))
    x_test = test_data['x']
    y_test = test_data['y'].astype('int32')

    n_y = int(y_test.max() + 1)
    y_transform = (lambda x: to_one_hot(x, n_y)) if one_hot else (lambda x: x)

    if isTf is True:
        x_train = x_train.transpose([0, 2, 3, 1])
        x_test = x_test.transpose([0, 2, 3, 1])

    return x_train, y_transform(y_train), x_test, y_transform(y_test)


def load_svhn(data_dir='/home/Data/', one_hot=False, isTf=True):
    data_dir = os.path.join(data_dir, 'svhn')
    train_dat = sio.loadmat(os.path.join(data_dir, 'train_32x32.mat'))
    train_x = train_dat['X'].astype('float32')
    train_y = train_dat['y'].flatten()
    train_y[train_y == 10] = 0
    train_x = train_x.transpose([3, 0, 1, 2])

    test_dat = sio.loadmat(os.path.join(data_dir, 'test_32x32.mat'))
    test_x = test_dat['X'].astype('float32')
    test_y = test_dat['y'].flatten()
    test_y[test_y == 10] = 0
    test_x = test_x.transpose([3, 0, 1, 2])

    n_y = int(train_y.max() + 1)
    y_transform = (lambda x: to_one_hot(x, n_y)) if one_hot else (lambda x: x)

    if isTf is False:
        train_x = train_x.transpose([0, 3, 1, 2])
        test_x = test_x.transpose([0, 3, 1, 2])

    train_x, test_x = train_x / 255., test_x / 255.
    return train_x, y_transform(train_y), test_x, y_transform(test_y)


def load_celebA(data_dir='/home/Data', num_dev=5000, isTf=True):
    data_dir = os.path.join(data_dir, 'celebA_64x64.npy')
    # rng_state = np.random.get_state()

    data = np.load(data_dir)
    np.random.shuffle(data)

    train_x = data[num_dev:]
    test_x = data[:num_dev]

    if isTf is True:
        train_x = train_x.transpose([0, 2, 3, 1])
        test_x = test_x.transpose([0, 2, 3, 1])

    train_x, test_x = train_x / 255., test_x / 255.
    train_x = train_x.astype(np.float32)
    test_x = test_x.astype(np.float32)
    return train_x, test_x


class DataSet(object):
    """Container class for a dataset (deprecated).

  THIS CLASS IS DEPRECATED. See
  [contrib/learn/README.md](https://www.tensorflow.org/code/tensorflow/contrib/learn/README.md)
  for general migration instructions.
  """

    def __init__(self,
                 images,
                 labels=None,
                 fake_data=False,
                 one_hot=False,
                 reshape=False,
                 seed=None):
        """Construct a DataSet.
    one_hot arg is used only if fake_data is true.  `dtype` can be either
    `uint8` to leave the input as `[0, 255]`, or `float32` to rescale into
    `[0, 1]`.  Seed arg provides for convenient deterministic testing.
    """
        import tensorflow as tf
        from tensorflow.python.framework import dtypes, random_seed

        if labels is None:
            labels = np.zeros([images.shape[0], 0])
        seed1, seed2 = random_seed.get_seed(seed)
        # If op level seed is not set, use whatever graph level seed is returned
        numpy.random.seed(seed1 if seed is None else seed2)
        dtype = dtypes.as_dtype(tf.float32).base_dtype
        if dtype not in (dtypes.uint8, dtypes.float32):
            raise TypeError(
                'Invalid image dtype %r, expected uint8 or float32' % dtype)
        if fake_data:
            self._num_examples = 10000
            self.one_hot = one_hot
        else:
            assert images.shape[0] == labels.shape[0], (
                'images.shape: %s labels.shape: %s' % (images.shape,
                                                       labels.shape))
            self._num_examples = images.shape[0]

            # Convert shape from [num examples, rows, columns, depth]
            # to [num examples, rows*columns] (assuming depth == 1)
            if reshape:
                images = images.reshape(
                    images.shape[0],
                    images.shape[1] * images.shape[2] * images.shape[3])
            if dtype == dtypes.float32:
                # Convert from [0, 255] -> [0.0, 1.0].
                if np.max(images) >= 1.01:
                    images = images.astype(numpy.float32)
                    images = numpy.multiply(images, 1.0 / 255.0)
        self._images = images
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size, fake_data=False, shuffle=True):
        """Return the next `batch_size` examples from this data set."""
        if fake_data:
            fake_image = [1] * 784
            if self.one_hot:
                fake_label = [1] + [0] * 9
            else:
                fake_label = 0
            return [fake_image for _ in range(batch_size)], [
                fake_label for _ in range(batch_size)
            ]
        start = self._index_in_epoch
        # Shuffle for the first epoch
        if self._epochs_completed == 0 and start == 0 and shuffle:
            perm0 = numpy.arange(self._num_examples)
            numpy.random.shuffle(perm0)
            self._images = self.images[perm0]
            self._labels = self.labels[perm0]
        # Go to the next epoch
        if start + batch_size > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Get the rest examples in this epoch
            rest_num_examples = self._num_examples - start
            images_rest_part = self._images[start:self._num_examples]
            labels_rest_part = self._labels[start:self._num_examples]
            # Shuffle the data
            if shuffle:
                perm = numpy.arange(self._num_examples)
                numpy.random.shuffle(perm)
                self._images = self.images[perm]
                self._labels = self.labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end = self._index_in_epoch
            images_new_part = self._images[start:end]
            labels_new_part = self._labels[start:end]
            return numpy.concatenate(
                (images_rest_part, images_new_part), axis=0), numpy.concatenate(
                    (labels_rest_part, labels_new_part), axis=0)
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self._images[start:end], self._labels[start:end]
