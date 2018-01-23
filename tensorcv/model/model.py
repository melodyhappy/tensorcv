import tensorflow as tf


class Model(object):
    def __init__(self, config):
        self.config = config

    def net(self, features, is_training):
        raise NotImplementedError()

    def loss(self, net_out, labels):
        raise NotImplementedError()

    def predictions(self, net_out):
        return {'predition': net_out}

    def metrics(self, net_out, label, mode):
        return {}

    def optimizer(self):
        raise NotImplementedError()

    def summary(self, features, labels, predictions, metrics, mode):
        pass


class ClassifierModel(Model):
    def predictions(self, net_out):
        prediction = tf.nn.softmax(net_out, name='prediction')
        return {'prediction': prediction}

    def loss(self, net_out, labels):
        loss = tf.losses.sparse_softmax_cross_entropy(
               labels=labels, logits=net_out)
        return loss

    def metrics(self, net_out, labels, mode):
        classes = tf.cast(tf.argmax(net_out, axis=1), tf.int32)
        if mode == tf.estimator.ModeKeys.TRAIN:
            correct = tf.cast(tf.equal(classes, labels), tf.float32)
            accuracy = tf.reduce_mean(correct)
        else:
            accuracy = tf.metrics.accuracy(labels, classes)
        metrics = {'accuracy': accuracy}
        return metrics

    def summary(self, features, labels, predictions, metrics, mode):
        tf.summary.image('image', features, 10)
        accuracy = metrics['accuracy']
        if mode == tf.estimator.ModeKeys.TRAIN:
            tf.summary.scalar('accuracy', accuracy)
        else:
            tf.summary.scalar('accuracy', accuracy[1])

