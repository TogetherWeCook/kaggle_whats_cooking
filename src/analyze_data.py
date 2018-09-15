#!/usr/bin/env python
# -*- coding: utf-8 -*-


import json
from itertools import chain
from collections import Counter

import pandas as pd

from config import train_data, test_data


__author__ = ['chaonan99']


def ingredient_compare():
    train_content = json.load(open(train_data))
    test_content = json.load(open(test_data))
    get_ingredients = lambda x: set(chain(*map(lambda x: x['ingredients'], x)))
    train_ingredients = get_ingredients(train_content)
    test_ingredients = get_ingredients(test_content)
    print(len(train_ingredients))
    print(len(test_ingredients))
    print(len(train_ingredients & test_ingredients))
    from IPython import embed; embed(); import os; os._exit(1)


def main():
    # content = json.load(open(train_data))
    # all_cuisine = set(map(lambda x: x['cuisine'], content))
    # all_ingredients = set(chain(*map(lambda x: x['ingredients'], content)))

    # ingredients_count = Counter(chain(*map(lambda x: x['ingredients'],
    #                                        content)))
    # df_train = pd.DataFrame(content)

    ingredient_compare()
    from IPython import embed; embed(); import os; os._exit(1)


if __name__ == '__main__':
    main()