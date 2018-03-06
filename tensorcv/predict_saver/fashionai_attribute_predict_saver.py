import os
from tqdm import tqdm
import tensorflow as tf

from tensorcv.predict_saver.predict_saver import PredictSaver
from tensorcv.logger import logger


class FashionaiAttributePredictSaver(PredictSaver):
    def save(self, predict_results, test_data, test_name):
        prediction_dir = os.path.join(
            self.config.eval_dir, str(self.config.model_step))
        prediction_path = os.path.join(prediction_dir, test_name + '.csv')
        params = self.config.predict_saver_params
        attribute = params['attribute']
        lines = []
        tf.gfile.MakeDirs(prediction_dir)
        with tf.gfile.Open(test_data) as f:
            for line in f:
                path, t, label = line.split(',')
                if t == attribute:
                    lines.append(path + ',' + t)
        logger.info('Do predictions for {}'.format(test_name))
        with open(prediction_path, 'w') as f:
            for line, predictions in tqdm(zip(lines, predict_results)):
                output_line = line
                scores = []
                for p in predictions['prediction']:
                    scores.append('{:.8f}'.format(p))
                output_line += ',' + ';'.join(scores)
                f.write(output_line + '\n')

