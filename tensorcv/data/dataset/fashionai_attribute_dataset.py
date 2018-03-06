import os
import random
import tensorflow as tf
from tensorcv.data.dataset import TSVDataset


class FashionaiAttributeDataset(TSVDataset):
    def read_lines(self, path, mode):
        params = self.config.dataset_params
        lines = []
        attribute = params['attribute']
        with tf.gfile.Open(path) as f:
            for line in f:
                path, t, label = line.split(',')
                if mode == tf.estimator.ModeKeys.PREDICT:
                    path = os.path.join(params['test_data_folder'], path)
                else:
                    path = os.path.join(params['train_data_folder'], path)
                if 'y' in label:
                    label = str(label.index('y'))
                else:
                    label = '-1'
                if t == attribute:
                    lines.append([path, label])
        is_training = mode == tf.estimator.ModeKeys.TRAIN
        if is_training:
            random.shuffle(lines)
        return lines

    def parse_fn(self, line, mode):
        image_path = line[0]
        label = tf.string_to_number(line[1], out_type=tf.int32)
        image = tf.read_file(image_path)
        image = tf.image.decode_jpeg(image, self.config.image_channels)
        image = tf.image.resize_images(
            image, (self.config.image_height, self.config.image_width))
        image.set_shape([
            self.config.image_height, self.config.image_width,
            self.config.image_channels
        ])
        image = tf.cast(image, dtype=tf.float32)

        return image, label

