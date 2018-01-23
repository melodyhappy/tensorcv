import tensorflow as tf
from tensorcv.data.dataset import TFRecordDataset


class Cifar10TFRecordDataset(TFRecordDataset):
    def parse_fn(self, record, mode):
        features = tf.parse_single_example(
            record,
            features={
                'image': tf.FixedLenFeature([], tf.string),
                'label': tf.FixedLenFeature([], tf.int64),
            })
        image = tf.decode_raw(features['image'], tf.uint8)
        image.set_shape([3 * 32 * 32])

        # Reshape from [depth * height * width] to [depth, height, width].
        image = tf.cast(
            tf.transpose(tf.reshape(image, [3, 32, 32]), [1, 2, 0]),
            tf.float32)

        if mode == tf.estimator.ModeKeys.TRAIN:
            image = tf.image.resize_image_with_crop_or_pad(
                image, 40, 40)
            image = tf.random_crop(image, [32, 32, 3])
            image = tf.image.random_flip_left_right(image)
        image = tf.image.per_image_standardization(image)
        label = tf.cast(features['label'], tf.int32)

        return image, label
