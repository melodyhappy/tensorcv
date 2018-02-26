import tensorflow as tf


def normal_predictions(net_out, params):
    return {'prediction': net_out}


def softmax_predictions(net_out, params):
    prediction = tf.nn.softmax(net_out, name='prediction')
    return {'prediction': prediction}


PREDICTIONS_MAP = {
    'normal': normal_predictions,
    'softmax': softmax_predictions,
}


def get_predictions_fn(config):
    if config.predictions not in PREDICTIONS_MAP:
        raise ValueError('{} is not a valid predictions type'.format(config.predictions))
    return PREDICTIONS_MAP[config.predictions]

