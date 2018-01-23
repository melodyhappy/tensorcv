import tensorflow as tf
from tensorcv.trainer.trainer import Trainer
from tensorcv.trainer.hooks import CheckpointPerfactSaverHook


class SimpleTrainer(Trainer):
    def get_model_fn(self, model):
        def model_fn(features, labels, mode):
            is_training = mode == tf.estimator.ModeKeys.TRAIN
            with tf.variable_scope('net'):
                net_out =  model.net(features, is_training)
            predictions = model.predictions(net_out)
            if mode == tf.estimator.ModeKeys.PREDICT:
                return tf.estimator.EstimatorSpec(
                    mode=mode,
                    predictions=predictions)

            loss = model.loss(net_out, labels)
            metrics = model.metrics(net_out, labels, mode)
            model.summary(features, labels, loss, metrics, mode)
            if mode == tf.estimator.ModeKeys.EVAL:
                return tf.estimator.EstimatorSpec(
                    mode=mode,
                    loss=loss,
                    eval_metric_ops=metrics)

            optimizer = model.optimizer()
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                train_op = optimizer.minimize(
                    loss=loss,
                    global_step=tf.train.get_global_step())

            training_chief_hooks = [
                CheckpointPerfactSaverHook(
                    self.config.model_dir,
                    save_steps=self.config.model_save_steps)
            ]

            return tf.estimator.EstimatorSpec(
                mode=mode,
                loss=loss,
                train_op=train_op,
                training_chief_hooks=training_chief_hooks)

        return model_fn
