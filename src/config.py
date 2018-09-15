#!/usr/bin/env python
# -*- coding: utf-8 -*-


import os


__author__ = ['chaonan99']


def get_dir_name():
    return os.path.dirname(os.path.abspath(__file__))


def relative_path(path):
    return os.path.join(get_dir_name(), path)


train_data = relative_path('../data/train.json')
test_data = relative_path('../data/test.json')


def main():
    pass


if __name__ == '__main__':
    main()