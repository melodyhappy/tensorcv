import logging
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

logger = logging.getLogger('tensorcv')
formatter = logging.Formatter('[%(asctime)s %(name)s %(filename)s:%(lineno)d %(levelname)s] %(message)s')
ch = logging.StreamHandler()
ch.setFormatter(formatter)
logger.addHandler(ch)
logger.setLevel(logging.DEBUG)
