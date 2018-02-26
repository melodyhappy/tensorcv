import os
import tensorflow as tf

from tensorcv.data.dataset import get_dataset
from tensorcv.train.model import Model
from tensorcv.train.trainer import get_trainer
from tensorcv.predict_saver import get_predict_saver


class Experiment(object):
    def __init__(self, config):
        self.config = config
        self.dataset = get_dataset(config)
        self.model = Model(config)
        self.trainer = get_trainer(config)

        run_config = self.trainer.get_run_config()
        model_fn = self.trainer.get_model_fn(self.model)
        self.estimator = tf.estimator.Estimator(
            model_fn=model_fn, config=run_config)

    def train(self):
        train_input_fn = self.dataset.get_train_input_fn()
        eval_input_fn = self.dataset.get_validation_input_fn()
        experiment = tf.contrib.learn.Experiment(
            self.estimator,
            train_input_fn,
            eval_input_fn,
            train_steps=self.config.max_steps,
            eval_steps=None,
            min_eval_frequency=1)
        experiment.train_and_evaluate()

    def predict(self):
        predict_saver = get_predict_saver(self.config)
        input_fn_map = self.dataset.get_test_input_fn_map()
        model_path = os.path.join(
            self.config.model_dir,
            'model.ckpt-' + str(self.config.model_step))
        for test_name, input_fn in input_fn_map.items():
            predict_results = self.estimator.predict(
                input_fn=input_fn, checkpoint_path=model_path)
            predict_saver.save(
                predict_results, self.config.test_data[test_name], test_name)

