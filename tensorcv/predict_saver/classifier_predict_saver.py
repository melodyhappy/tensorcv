import os
from tqdm import tqdm
import tensorflow as tf

from tensorcv.predict_saver.predict_saver import PredictSaver
from tensorcv.logger import logger


class ClassifierTSVPredictSaver(PredictSaver):
    def save(self, predict_results, test_data, test_name):
        prediction_dir = os.path.join(
            self.config.eval_dir, str(self.config.model_step), 'prediction')
        prediction_path = os.path.join(prediction_dir, test_name + '.tsv')
        if tf.gfile.Exists(prediction_path) and not self.config.overwrite:
            logger.warning('{} exists.'.format(prediction_path))
            return
        tf.gfile.MakeDirs(prediction_dir)
        with tf.gfile.Open(test_data) as f:
            lines = [line.strip() for line in f]
            lines = [line for line in lines if line]
        logger.info('Do predictions for {}'.format(test_name))
        with open(prediction_path, 'w') as f:
            for line, predictions in tqdm(zip(lines, predict_results)):
                output_line = line
                for p in predictions['prediction']:
                    output_line += '\t{:.8f}'.format(p)
                f.write(output_line + '\n')

