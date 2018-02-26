import functools
import numpy as np
from PIL import Image, ImageDraw
import tensorflow as tf


def add_trainable_variables_histogram():
    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name, var)


def normal_summary(features, labels, predictions, mode, params):
    tf.summary.image('image', features, 10)
    add_trainable_variables_histogram()


def alignment_summary(features, labels, predictions, mode, params):
    image = tf.cast(features, tf.uint8)
    prediction = predictions['prediction']
    label_points = tf.stack([labels[:, ::2], labels[:, 1::2]], axis=2)
    predict_points = tf.stack(
        [prediction[:, ::2], prediction[:, 1::2]], axis=2)
    def draw_points(args):
        image, label_points, predict_points = args
        def draw_points_pyfn(image, points, color, radius=1):
            if image.shape[-1] == 1:
                image = np.squeeze(image, axis=-1)
            image_pil = Image.fromarray(np.uint8(image)).convert('RGB')
            draw = ImageDraw.Draw(image_pil)
            im_width, im_height = image_pil.size
            for point in points:
                x = point[0] * im_width
                y = point[1] * im_height
                draw.ellipse([(x - radius, y - radius),
                              (x + radius, y + radius)],
                             outline=color, fill=color)
            image = np.array(image_pil)
            return image
        image = tf.py_func(
            functools.partial(draw_points_pyfn, color=(0, 255, 0)),
            (image, label_points), tf.uint8)
        image = tf.py_func(
            functools.partial(draw_points_pyfn, color=(255, 0, 0)),
            (image, predict_points), tf.uint8)
        return image
    image = tf.map_fn(
        draw_points, (image, label_points, predict_points),
        dtype=tf.uint8, back_prop=False)
    tf.summary.image('image', image, 10)
    add_trainable_variables_histogram()


SUMMARY_MAP = {
    'normal': normal_summary,
    'alignment': alignment_summary,
}


def get_summary_fn(config):
    if config.summary not in SUMMARY_MAP:
        raise ValueError('{} is not a valid metrics type'.format(config.summary))
    return SUMMARY_MAP[config.summary]

