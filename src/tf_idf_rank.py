#!/usr/bin/env python
# -*- coding: utf-8 -*-


import json

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score


__author__ = ['chaonan99']


train_data = '../input/train.json'
test_data = '../input/test.json'


def run_pipeline(submission=False):
    np.random.seed(42)
    get_ingredients = lambda x: list(map(lambda x: x['ingredients'], x))
    get_cuisine = lambda x: list(map(lambda x: x['cuisine'], x))
    ingredient_format = lambda x: ' '.join(x)
    # ingredient_format = lambda x: ' '.join([i.replace(' ', '_') for i in x])

    train_content = json.load(open(train_data))
    test_content = json.load(open(test_data))
    np.random.shuffle(train_content)
    if not submission:
        train_content, dev_content = train_test_split(train_content,
                                                      test_size=0.25)

    tfidf = TfidfVectorizer(binary=True)
    X_tr = tfidf.fit_transform(map(ingredient_format,
                                   get_ingredients(train_content)))
    X_te = tfidf.transform(map(ingredient_format,
                               get_ingredients(test_content)))
    if not submission:
        X_dev = tfidf.transform(map(ingredient_format,
                                    get_ingredients(dev_content)))

    ## Nearest neighbor
    le = preprocessing.LabelEncoder()
    y_tr = le.fit_transform(get_cuisine(train_content))
    if not submission:
        y_dev = le.transform(get_cuisine(dev_content))

    # clf = svm.SVC(C=1000, kernel='rbf', verbose=True)
    # clf = svm.SVC(C=1000, kernel='rbf', class_weight='balanced', verbose=True)
    clf = svm.SVC(C=1000,
                  kernel='rbf',
                  # coef0=1,
                  # class_weight=None,
                  verbose=True)
    clf = OneVsRestClassifier(clf)
    clf.fit(X_tr, y_tr)

    if not submission:
        y_dev_pred = clf.predict(X_dev)
        print("Accuracy:", accuracy_score(y_dev, y_dev_pred))
    else:
        y_te = clf.predict(X_te)
        test_label_pred = le.inverse_transform(y_te)

        test_id = list(map(lambda x: x['id'], test_content))
        sub = pd.DataFrame({'id': test_id, 'cuisine': test_label_pred},
                           columns=['id', 'cuisine'])
        sub.to_csv('../dump/svm_output.csv', index=False)


def main():
    run_pipeline(submission=False)


if __name__ == '__main__':
    main()