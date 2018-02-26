import tensorflow as tf


def softmax_cross_entropy(label, logits, params):
    return tf.losses.softmax_cross_entropy(label, logits)


def sparse_softmax_cross_entropy(label, logits, params):
    return tf.losses.sparse_softmax_cross_entropy(label, logits)


def sigmoid_cross_entropy(label, logits, params):
    return tf.losses.sigmoid_cross_entropy(label, logits)


def mean_squared_error(label, logits, params):
    return tf.losses.mean_squared_error(label, logits)

LOSS_MAP = {
    'softmax_cross_entropy': softmax_cross_entropy,
    'sparse_softmax_cross_entropy': sparse_softmax_cross_entropy,
    'sigmoid_cross_entropy': sigmoid_cross_entropy,
    'mean_squared_error': mean_squared_error
}


def get_loss_fn(config):
    if config.loss not in LOSS_MAP:
        raise ValueError('{} is not a valid loss type'.format(config.loss))
    return LOSS_MAP[config.loss]

