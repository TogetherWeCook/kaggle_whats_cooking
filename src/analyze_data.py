#!/usr/bin/env python
# -*- coding: utf-8 -*-


import json
from itertools import chain
from collections import Counter

import pandas as pd

from config import train_data


__author__ = ['chaonan99']


def main():
    content = json.load(open(train_data))
    all_cuisine = set(map(lambda x: x['cuisine'], content))
    all_ingredients = set(chain(*map(lambda x: x['ingredients'], content)))

    ingredients_count = Counter(chain(*map(lambda x: x['ingredients'],
                                           content)))
    # df_train = pd.DataFrame(content)


if __name__ == '__main__':
    main()