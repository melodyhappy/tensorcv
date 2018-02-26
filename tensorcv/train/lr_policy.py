import tensorflow as tf


def fixed(global_step, params):
    assert 'base_lr' in params, 'base_lr must in params'
    lr = tf.constant(params['base_lr'])
    tf.summary.scalar('learining_rate', lr)
    return lr


def exponential_decay(global_step, params):
    assert 'base_lr' in params, 'base_lr must in params'
    assert 'decay_steps' in params, 'decay_steps must in params'
    assert 'decay_rate' in params, 'decay_rate must in params'

    lr = tf.train.exponential_decay(
        learning_rate=params['base_lr'],
        global_step=global_step,
        decay_steps=params['decay_steps'],
        decay_rate=params['decay_rate'],
        staircase=params.get('staircase', True),
        name='learning_rate')

    tf.summary.scalar('learining_rate', lr)
    return lr


def polynomial_decay(global_step, params):
    assert 'base_lr' in params, 'base_lr must in params'
    assert 'decay_steps' in params, 'decay_steps must in params'
    assert 'end_learning_rate' in params, 'end_learning_rate must in params'
    assert 'power' in params, 'power must in params'

    lr = tf.train.exponential_decay(
        learning_rate=params['base_lr'],
        global_step=global_step,
        decay_steps=params['decay_steps'],
        end_learning_rate=params['end_learning_rate'],
        power=params['power'],
        name='learning_rate')

    tf.summary.scalar('learining_rate', lr)
    return lr


LR_POLICY_MAP = {
    'fixed': fixed,
    'exponential_decay': exponential_decay,
    'polynomial_decay': polynomial_decay,
}


def get_lr_policy_fn(config):
    if config.lr_policy not in LR_POLICY_MAP:
        raise ValueError('{} is not a valid lr policy type'.format(config.lr_policy))
    return LR_POLICY_MAP[config.lr_policy]

