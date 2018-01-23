#!/usr/bin/env python
# coding: utf-8

import os
import click

from tensorcv.experiment import Experiment
from tensorcv.experiment import ExperimentSpec

__all__ = ['predict']


@click.command(help='Get prediction result of a model')
@click.argument('config')
@click.option('--gpu', '-g', default='0', help='The GPU list.')
def predict(config, gpu):
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu

    config = ExperimentSpec(config)
    config.num_gpus = len([g for g in gpu.split(',') if g])

    exp = Experiment(config)
    exp.predict()


if __name__ == '__main__':
    predict()
