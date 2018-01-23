import tensorflow as tf
from tensorcv.data.dataset.dataset import TSVDataset
from tensorcv.data.dataset.dataset import TFRecordDataset


class ClassifierTSVDataset(TSVDataset):
    def parse_fn(self, line, mode):
        image_path = line[0]
        label = tf.string_to_number(line[1], out_type=tf.int32)
        image = tf.read_file(image_path)
        image = tf.image.decode_image(image, self.config.image_channels)
        image.set_shape([
            self.config.image_height, self.config.image_width,
            self.config.image_channels
        ])
        image = tf.cast(image, dtype=tf.float32)
        return image, label


class ClassfierTFRecordDataset(TFRecordDataset):
    # TODO
    pass

