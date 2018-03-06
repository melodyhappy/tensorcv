from tensorcv.data.dataset.dataset import Dataset
from tensorcv.data.dataset.dataset import TSVDataset
from tensorcv.data.dataset.dataset import TFRecordDataset
from tensorcv.data.dataset.classifier_dataset import ClassifierTSVDataset
from tensorcv.data.dataset.cifar10_dataset import Cifar10TFRecordDataset
from tensorcv.data.dataset.fashionai_attribute_dataset import FashionaiAttributeDataset


DATASET_MAP = {
    'classifiertsvdataset': ClassifierTSVDataset,
    'cifar10tfrecorddataset': Cifar10TFRecordDataset,
    'fashionaiattributedataset': FashionaiAttributeDataset
}


def get_dataset(config):
    dataset_type = config.dataset_type.lower()
    if dataset_type not in DATASET_MAP:
        raise ValueError('{} is not a valid data type'.format(dataset_type))
    return DATASET_MAP[dataset_type](config)

