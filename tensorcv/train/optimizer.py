import tensorflow as tf


def gradient_descent_optimizer(learning_rate, params):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    return optimizer


def momentum_optimizer(learning_rate, params):
    assert 'momentum' in params, 'momentum must in params'
    optimizer = tf.train.MomentumOptimizer(
        learning_rate, params['momentum'])
    return optimizer


def adam_optimizer(learning_rate, params):
    optimizer = tf.train.AdamOptimizer(
        learning_rate=learning_rate,
        beta1=params.get('beta1', 0.9),
        beta2=params.get('beta2', 0.999),
        epsilon=params.get('epsilon', le-8)
    )
    return optimizer


OPTIMIZER_MAP = {
    'gradient_descent': gradient_descent_optimizer,
    'momentum': momentum_optimizer,
    'adam': adam_optimizer
}


def get_optimizer_fn(config):
    if config.optimizer not in OPTIMIZER_MAP:
        raise ValueError('{} is not a valid optimizer type'.format(config.optimizer))
    return OPTIMIZER_MAP[config.optimizer]

