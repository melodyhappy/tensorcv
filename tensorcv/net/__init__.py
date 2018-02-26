from tensorcv.net.resnet import resnet_v2
from tensorcv.net.resnet import cifar10_resnet_v2


NET_MAP = {
    'resnet_v2': resnet_v2,
    'cifar10_resnet_v2': cifar10_resnet_v2
}


def get_net_fn(config):
    if config.net not in NET_MAP:
        raise ValueError('{} is not a valid net type'.format(config.net))
    return NET_MAP[config.net]
