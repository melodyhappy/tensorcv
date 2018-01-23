import tensorflow as tf
from tensorcv.model import ClassifierModel
from tensorcv.net import resnet


class Cifar10Model(ClassifierModel):
    def net(self, features, is_training):
        network = resnet.cifar10_resnet_v2_generator(32, 10)
        x = network(features, is_training)
        return x

    def optimizer(self):
        lr = tf.train.exponential_decay(
            learning_rate=1e-3,
            global_step=tf.train.get_global_step(),
            decay_steps=5000,
            decay_rate=0.75,
            staircase=True,
            name='learning_rate')
        tf.summary.scalar('lr', lr)

        optimizer = tf.train.MomentumOptimizer(
            learning_rate=lr,
            momentum=0.9)
        return optimizer

