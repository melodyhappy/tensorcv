import tensorflow as tf
from tensorcv.train.trainer.trainer import Trainer
from tensorcv.train.hooks import CheckpointPerfactSaverHook


class MultiGPUTrainer(Trainer):
    def feature_shard(self, feature, num_shards):
        if num_shards > 1:
            feature_batch = tf.unstack(
                feature, num=self.config.batch_size, axis=0)
            feature_shards = [[] for i in range(num_shards)]
            for i in range(self.config.batch_size):
                idx = i % num_shards
                feature_shards[idx].append(feature_batch[i])
            feature_shards = [tf.parallel_stack(x) for x in feature_shards]
        else:
            feature_shards = [feature]
        return feature_shards

    def average_gradients(self, tower_grads):
        average_grads = []
        for grad_and_vars in zip(*tower_grads):
            grads = []
            for g, _ in grad_and_vars:
                expanded_g = tf.expand_dims(g, 0)
                grads.append(expanded_g)
            grad = tf.concat(axis=0, values=grads)
            grad = tf.reduce_mean(grad, 0)
            v = grad_and_vars[0][1]
            grad_and_var = (grad, v)
            average_grads.append(grad_and_var)
        return average_grads

    def get_model_fn(self, model):
        def model_fn(features, labels, mode):
            is_training = mode == tf.estimator.ModeKeys.TRAIN

            if mode == tf.estimator.ModeKeys.PREDICT:
                with tf.variable_scope('net'):
                    net_out =  model.net(features, is_training)
                predictions = model.predictions(net_out)
                return tf.estimator.EstimatorSpec(
                    mode=mode,
                    predictions=predictions)

            if mode == tf.estimator.ModeKeys.EVAL:
                with tf.variable_scope('net'):
                    net_out =  model.net(features, is_training)
                loss = model.loss(labels, net_out)
                predictions = model.predictions(net_out)
                metrics = model.metrics(labels, net_out, mode)
                model.summary(features, labels, predictions, mode)
                return tf.estimator.EstimatorSpec(
                    mode=mode,
                    loss=loss,
                    eval_metric_ops=metrics)

            assert mode == tf.estimator.ModeKeys.TRAIN

            num_gpus = self.config.num_gpus
            feature_shards = self.feature_shard(features, num_gpus)
            label_shards = self.feature_shard(labels, num_gpus)

            lr = model.lr_policy(tf.train.get_global_step())
            optimizer = model.optimizer(lr)
            tower_losses = []
            tower_grads = []
            for i in range(num_gpus):
                with tf.variable_scope('net', reuse=bool(i != 0)):
                    with tf.name_scope('tower_%d' % i) as name_scope:
                        with tf.device('/gpu:%d' % i):
                            feature = feature_shards[i]
                            label = label_shards[i]
                            net_out = model.net(feature, is_training)
                            loss = model.loss(label, net_out)
                            grads = optimizer.compute_gradients(loss)
                            tower_losses.append(loss)
                            tower_grads.append(grads)
                            if i == 0:
                                update_ops =  tf.get_collection(
                                    tf.GraphKeys.UPDATE_OPS, name_scope)
                                net_out_0 = net_out
                                feature_0 = feature
                                label_0 = label
            metrics = model.metrics(label_0, net_out_0, mode)
            grads = self.average_gradients(tower_grads)
            loss = tf.reduce_mean(tower_losses)
            apply_gradient_op = optimizer.apply_gradients(
                grads, global_step=tf.train.get_global_step())
            train_op = [apply_gradient_op]
            train_op.extend(update_ops)
            train_op = tf.group(*train_op)

            predictions_0 = model.predictions(net_out_0)
            model.summary(feature_0, label_0, predictions_0, mode)

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

            training_chief_hooks = [
                CheckpointPerfactSaverHook(
                    self.config.model_dir,
                    save_steps=self.config.model_save_steps)
            ]

            return tf.estimator.EstimatorSpec(
                mode=mode,
                train_op=train_op,
                loss=loss,
                training_chief_hooks=training_chief_hooks)

        return model_fn

