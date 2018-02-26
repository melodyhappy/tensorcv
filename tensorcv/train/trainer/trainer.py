import tensorflow as tf


class Trainer(object):
    def __init__(self, config):
        self.config = config

    def get_run_config(self):
        run_config = tf.estimator.RunConfig(
            model_dir=self.config.model_dir,
            save_summary_steps=self.config.summary_steps,
            keep_checkpoint_max=None)
        return run_config

    def get_model_fn(self, model):
        raise NotImplementedError()

