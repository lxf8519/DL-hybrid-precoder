# Paper: Deep Learning for Direct Hybrid Precoding in Millimeter Wave Massive MIMO Systems  https://arxiv.org/abs/1905.13212
import numpy as np
import scipy.io as scio
import h5py

def dense_to_one_hot(labels_dense, num_classes):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    labels_one_hot = np.zeros((num_labels, num_classes))
    for i in range(num_labels):
        labels_one_hot[i][int(labels_dense[i])] = 1
    return labels_one_hot


class DataSet(object):
    def __init__(self,
                 data,
                 labels
                 ):
        """Construct a DataSet."""
        self._num_examples = data.shape[0]
        self._data = data
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def data(self):
        return self._data

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size, shuffle=True):
        """Return the next `batch_size` examples from this data set."""

        start = self._index_in_epoch
        # Shuffle for the first epoch
        if self._epochs_completed == 0 and start == 0 and shuffle:
            perm0 = np.arange(self._num_examples)
            np.random.shuffle(perm0)
            self._data = self.data[perm0]
            self._labels = self.labels[perm0]
        # Go to the next epoch
        if start + batch_size > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Get the rest examples in this epoch
            rest_num_examples = self._num_examples - start
            data_rest_part = self._data[start:self._num_examples]
            labels_rest_part = self._labels[start:self._num_examples]
            # Shuffle the data
            if shuffle:
                perm = np.arange(self._num_examples)
                np.random.shuffle(perm)
                self._data = self.data[perm]
                self._labels = self.labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size - rest_num_examples
            end = self._index_in_epoch
            data_new_part = self._data[start:end]
            labels_new_part = self._labels[start:end]
            return np.concatenate((data_rest_part, data_new_part), axis=0), np.concatenate(
                (labels_rest_part, labels_new_part), axis=0)
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self._data[start:end], self._labels[start:end]



def read_data_sets(filename_db_tr = 'MIMO_dataset/DeepMIMO_dataset_ch_only_concat_global_BS16x_MS16_train.mat',
                   filename_db_test='MIMO_dataset/DeepMIMO_dataset_ch_only_concat_global_BS16x_MS16_test_beam_labels.mat',
                   filename_db_tr_label = 'MIMO_dataset/DeepMIMO_dataset_ch_only_concat_global_BS16x_MS16_train_beam_labels.mat',
                   filename_db_test_label = 'MIMO_dataset/DeepMIMO_dataset_ch_only_concat_global_BS16x_MS16_test_beam_labels.mat',
                   HDF5file=True):
    """Flag HDF5file is True if mat files from MATLAB is stored in hd5 format,
	H_train, H_test, train_labels, test_labels are the names of the variables saved in MATLAB"""
    if not HDF5file:
        rtmp = scio.loadmat(filename_db_tr)
        train_data = rtmp['H_train']
        rtmp = scio.loadmat(filename_db_test)
        test_data = rtmp['H_test']
        test_data = rtmp['H_test']
        rtmp = scio.loadmat(filename_db_tr_label)
        train_data_label = rtmp['train_labels']
        rtmp = scio.loadmat(filename_db_test_label)
        test_data_label = rtmp['test_labels']
    else:
        file = h5py.File(filename_db_tr)
        train_data = np.array(file['H_train']).transpose()
        file = h5py.File(filename_db_test)
        test_data = np.array(file['H_test']).transpose()
        file = h5py.File(filename_db_tr_label)
        train_data_label = np.array(file['train_labels']).transpose()
        file = h5py.File(filename_db_test_label)
        test_data_label = np.array(file['test_labels']).transpose()

    num_samples = train_data.shape[0]
    #Use dense_to_one_hot if labels are not one-hot vectors
    train_labels=train_data_label


    print('Dataset size is %d' % (num_samples))
    num_samples_test = test_data.shape[0]
    #Use dense_to_one_hot if labels are not one-hot vectors
    test_labels=test_data_label

    train = DataSet(train_data, train_labels)
    validation = DataSet(test_data, test_labels)

    return train, validation