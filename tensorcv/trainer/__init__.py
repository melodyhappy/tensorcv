from tensorcv.trainer.simple_trainer import SimpleTrainer
from tensorcv.trainer.multi_gpu_trainer import MultiGPUTrainer


def get_trainer(config):
    if config.num_gpus == 1:
        return SimpleTrainer(config)
    else:
        return MultiGPUTrainer(config)
