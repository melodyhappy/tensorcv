#!/bin/bash

if [[ ! -e /tmp/tensorcv-jobs/cifar10/data ]]; then
    python2 generate_cifar10_tfrecords.py --data-dir /tmp/tensorcv-jobs/cifar10/data
fi

tcv train cifar10.cfg --gpu 0
