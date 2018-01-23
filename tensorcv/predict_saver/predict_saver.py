class PredictSaver(object):
    def __init__(self, config):
        self.config = config

    def save(self, predict_results, test_data, test_name):
        raise NotImplementedError()
