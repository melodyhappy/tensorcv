from tensorcv.predict_saver.classifier_predict_saver import ClassifierTSVPredictSaver
from tensorcv.predict_saver.fashionai_attribute_predict_saver import FashionaiAttributePredictSaver


PREDICT_SAVER_MAP = {
    'classifiertsvpredictsaver': ClassifierTSVPredictSaver,
    'fashionaiattributepredictsaver': FashionaiAttributePredictSaver
}


def get_predict_saver(config):
    predict_saver_type = config.predict_saver_type.lower()
    if predict_saver_type == 'customize':
        return config.get_predict_saver()(config)
    if predict_saver_type not in PREDICT_SAVER_MAP:
        raise ValueError('{} is not valid a evaluator'.format(predict_saver_type))
    return PREDICT_SAVER_MAP[predict_saver_type](config)
