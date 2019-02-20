import numpy as np


class DataLoader(object):

    def __init__(self, data, label, batch_size=100, rng=None, shuffle=False,
                 return_labels=False):
        """
        - data_dir is location where to store files
        - subset is train|test
        - batch_size is int, of #examples to load at once
        - rng is np.random.RandomState object for reproducibility
        """

        self.data = data.astype(np.float32)
        if label is None:
            label = np.zeros((self.data.shape[0], 0))
        self.labels = label

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.return_labels = return_labels

        self.p = 0  # pointer to where we are in iteration
        self.rng = np.random.RandomState(1) if rng is None else rng

    def get_observation_size(self):
        return self.data.shape[1:]

    def get_num_labels(self):
        return np.amax(self.labels) + 1

    def reset(self):
        self.p = 0

    def __iter__(self):
        return self

    def __next__(self, n=None):
        """ n is the number of examples to fetch """
        n = self.batch_size if n is None else n

        # on first iteration lazily permute all data
        if self.p == 0 and self.shuffle:
            inds = self.rng.permutation(self.data.shape[0])
            self.data = self.data[inds]
            self.labels = self.labels[inds]

        # on last iteration reset the counter and raise StopIteration
        if self.p + n > self.data.shape[0]:
            self.reset()  # reset for next time we get called
            raise StopIteration

        # on intermediate iterations fetch the next batch
        x = self.data[self.p: self.p + n]
        y = self.labels[self.p: self.p + n]
        self.p += self.batch_size

        if self.return_labels:
            return x, y
        else:
            return x

    next = __next__
