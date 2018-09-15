#!/usr/bin/env python
# -*- coding: utf-8 -*-


import json

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm

from config import train_data, test_data


__author__ = ['chaonan99']


def main():
    np.random.seed(42)
    get_ingredients = lambda x: list(map(lambda x: x['ingredients'], x))
    get_cuisine = lambda x: list(map(lambda x: x['cuisine'], x))
    ingredient_format = lambda x: ' '.join([i.replace(' ', '_') for i in x])

    train_content = json.load(open(train_data))
    test_content = json.load(open(test_data))
    np.random.shuffle(train_content)
    train_content, dev_content = train_test_split(train_content)

    tfidf = TfidfVectorizer()
    X_tr = tfidf.fit_transform(map(ingredient_format, get_ingredients(train_content)))
    X_dev = tfidf.transform(map(ingredient_format, get_ingredients(dev_content)))
    # X_te = tfidf.transform(map(ingredient_format, get_ingredients(test_content)))

    ## Nearest neighbor
    le = preprocessing.LabelEncoder()
    y_tr = le.fit_transform(get_cuisine(train_content))
    y_dev = le.transform(get_cuisine(dev_content))

    from IPython import embed; embed(); import os; os._exit(1)
    clf = svm.SVC(C=100, kernel='rbf', verbose=True)
    clf.fit(X_tr, y_tr)
    clf.predict(X_dev)


if __name__ == '__main__':
    main()