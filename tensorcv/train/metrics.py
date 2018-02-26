import tensorflow as tf


def normal_metrics(labels, net_out, mode, params):
    return {}


def accuracy_metrics(labels, net_out, mode, params):
    classes = tf.argmax(net_out, axis=1)
    classes = tf.cast(classes, dtype=tf.int32)
    if mode == tf.estimator.ModeKeys.TRAIN:
        correct = tf.cast(tf.equal(classes, labels), tf.float32)
        accuracy = tf.reduce_mean(correct)
        tf.summary.scalar('accuracy', accuracy)
    else:
        accuracy = tf.metrics.accuracy(labels, classes)
        tf.summary.scalar('accuracy', accuracy[1])
    metrics = {'accuracy': accuracy}
    return metrics


METRICS_MAP = {
    'normal': normal_metrics,
    'accuracy': accuracy_metrics,
}


def get_metrics_fn(config):
    if config.metrics not in METRICS_MAP:
        raise ValueError('{} is not a valid metrics type'.format(config.metrics))
    return METRICS_MAP[config.metrics]

