from tensorcv.model.model import ClassifierModel
from tensorcv.model.cifar10_model import Cifar10Model


MODEL_MAP = {
    'clasifiermodel': ClassifierModel,
    'cifar10model': Cifar10Model
}


def get_model(config):
    model_type = config.model_type.lower()
    if model_type not in MODEL_MAP:
        raise ValueError('{} is not a valid data type'.format(model_type))
    return MODEL_MAP[model_type](config)

