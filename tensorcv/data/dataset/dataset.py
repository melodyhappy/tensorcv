import random
import functools
import tensorflow as tf


class Dataset(object):
    def __init__(self, config):
        self.config = config

    def get_input_fn(self, path, mode):
        raise NotImplementedError()

    def get_train_input_fn(self):
        input_fn = self.get_input_fn(
            self.config.train_data, tf.estimator.ModeKeys.TRAIN)
        return input_fn

    def get_validation_input_fn(self):
        input_fn = self.get_input_fn(
            self.config.validation_data, tf.estimator.ModeKeys.EVAL)
        return input_fn

    def get_test_input_fn_map(self, mode=tf.estimator.ModeKeys.PREDICT):
        input_fn_map = {}
        for name, path in self.config.test_data.items():
            input_fn = self.get_input_fn(path, mode)
            input_fn_map[name] = input_fn
        return input_fn_map


class TSVDataset(Dataset):
    def parse_fn(self, line, mode):
        raise NotImplementedError()

    def read_lines(self, path, mode):
        with tf.gfile.Open(path) as f:
            lines = [line.strip() for line in f]
            lines = [line.split('\t') for line in lines if line]
        is_training = mode == tf.estimator.ModeKeys.TRAIN
        if is_training:
            random.shuffle(lines)
        return lines

    def get_input_fn(self, path, mode):
        lines = self.read_lines(path, mode)
        size = len(lines)
        is_training = mode == tf.estimator.ModeKeys.TRAIN
        dataset = tf.data.Dataset.from_tensor_slices(lines)
        if is_training:
            dataset = dataset.shuffle(
                buffer_size=self.config.shuffle_buffer_size)
            dataset = dataset.repeat()
            dataset = dataset.map(
                functools.partial(self.parse_fn, mode=mode),
                num_parallel_calls=self.config.num_data_processes)
            dataset = dataset.batch(self.config.batch_size)
            dataset = dataset.prefetch(self.config.prefetch_batches)
        else:
            dataset = dataset.map(
                functools.partial(self.parse_fn, mode=mode),
                num_parallel_calls=self.config.num_data_processes)
            dataset = dataset.batch(self.config.batch_size)
            dataset = dataset.prefetch(self.config.prefetch_batches)
        def input_fn():
            iterator = dataset.make_one_shot_iterator()
            features, labels = iterator.get_next()
            return features, labels
        return input_fn


class TFRecordDataset(Dataset):
    def parse_fn(self, record, mode):
        raise NotImplementedError()

    def get_input_fn(self, path, mode):
        is_training = mode == tf.estimator.ModeKeys.TRAIN
        dataset = tf.data.TFRecordDataset(path)
        dataset = dataset.map(
            functools.partial(self.parse_fn, mode=mode),
            num_parallel_calls=self.config.num_data_processes)
        dataset = dataset.prefetch(
            self.config.prefetch_batches * self.config.batch_size)
        if is_training:
            dataset = dataset.shuffle(
                buffer_size=self.config.shuffle_buffer_size)
            dataset = dataset.repeat()
        dataset = dataset.batch(self.config.batch_size)
        dataset = dataset.prefetch(self.config.prefetch_batches)
        def input_fn():
            iterator = dataset.make_one_shot_iterator()
            features, labels = iterator.get_next()
            return features, labels
        return input_fn

