import os
import json
from configparser import ConfigParser, ExtendedInterpolation

class ExperimentSpec(object):
    def __init__(self, config_path):
        assert os.path.exists(config_path), '{} not exists.'.format(config_path)
        self.config = ConfigParser(
            delimiters='=',
            interpolation=ExtendedInterpolation())
        self.config.read(config_path)

    @property
    def exp_dir(self):
        return self.config.get('env', 'exp_dir')

    @property
    def data_dir(self):
        default = os.path.join(self.exp_dir, 'data')
        return self.config.get('env', 'data_dir', fallback=default)

    @property
    def log_dir(self):
        default = os.path.join(self.exp_dir, 'log')
        return self.config.get('env', 'log_dir', fallback=default)

    @property
    def model_dir(self):
        default = os.path.join(self.exp_dir, 'model')
        return self.config.get('env', 'model_dir', fallback=default)

    @property
    def eval_dir(self):
        default = os.path.join(self.exp_dir, 'eval')
        return self.config.get('env', 'eval_dir', fallback=default)

    @property
    def train_data(self):
        return self.config.get('data', 'train_data', fallback='')

    @property
    def validation_data(self):
        return self.config.get('data', 'validation_data', fallback='')

    @property
    def test_data(self):
        raw_string = self.config.get('data', 'test_data', fallback='')
        test_data = {}
        for testset in raw_string.split('\n'):
            if testset:
                name, path = testset.split()
                test_data[name] = path
        return test_data

    @property
    def image_height(self):
        return self.config.getint('data', 'image_height')

    @property
    def image_width(self):
        return self.config.getint('data', 'image_width')

    @property
    def image_channels(self):
        return self.config.getint('data', 'image_channels')

    @property
    def image_format(self):
        return self.config.get('data', 'image_format', fallback='jpeg')

    @property
    def batch_size(self):
        return self.config.getint('data', 'batch_size')

    @property
    def shuffle_buffer_size(self):
        return self.config.getint('data', 'shuffle_buffer_size', fallback=10000)

    @property
    def prefetch_batches(self):
        return self.config.getint('data', 'prefetch_batches', fallback=20)

    @property
    def num_data_processes(self):
        return self.config.getint('data', 'num_data_processes', fallback=10)

    @property
    def dataset_type(self):
        return self.config.get('data', 'dataset_type')

    @property
    def net(self):
        return self.config.get('train', 'net')

    @property
    def net_params(self):
        params = self.config.get('train', 'net_params', fallback='{}')
        return json.loads(params)

    @property
    def loss(self):
        return self.config.get('train', 'loss')

    @property
    def loss_params(self):
        params = self.config.get('train', 'loss_params', fallback='{}')
        return json.loads(params)

    @property
    def predictions(self):
        return self.config.get('train', 'predictions')

    @property
    def predictions_params(self):
        params = self.config.get('train', 'predictions_params', fallback='{}')
        return json.loads(params)

    @property
    def metrics(self):
        return self.config.get('train', 'metrics')

    @property
    def metrics_params(self):
        params = self.config.get('train', 'metrics_params', fallback='{}')
        return json.loads(params)

    @property
    def lr_policy(self):
        return self.config.get('train', 'lr_policy')

    @property
    def lr_policy_params(self):
        params = self.config.get('train', 'lr_policy_params', fallback='{}')
        return json.loads(params)

    @property
    def optimizer(self):
        return self.config.get('train', 'optimizer')

    @property
    def optimizer_params(self):
        params = self.config.get('train', 'optimizer_params', fallback='{}')
        return json.loads(params)

    @property
    def summary(self):
        return self.config.get('train', 'summary')

    @property
    def summary_params(self):
        params = self.config.get('train', 'summary_params', fallback='{}')
        return json.loads(params)

    @property
    def max_steps(self):
        return self.config.getint('train', 'max_step', fallback=None)

    @property
    def summary_steps(self):
        return self.config.getint('train', 'summary_steps', fallback=100)

    @property
    def model_save_steps(self):
        return self.config.getint('train', 'model_save_steps', fallback=1000)

    @property
    def predict_saver_type(self):
        return self.config.get('evaluate', 'predict_saver_type')

