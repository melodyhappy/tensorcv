#!/usr/bin/env python
# coding: utf-8

import click

from tensorcv.command.aliased_group import AliasedGroup
from tensorcv.command.train import train
from tensorcv.command.predict import predict

__all__ = ['cli']


CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])


@click.command(cls=AliasedGroup, context_settings=CONTEXT_SETTINGS)
def cli():
    pass


cli.add_command(train)
cli.add_command(predict)


if __name__ == '__main__':
    cli()
