#!/bin/bash

python2 generate_cifar10_tfrecords.py --data-dir /tmp/tensorcv-jobs/cifar10/data

tcv train cifar10.cfg --gpu 0
