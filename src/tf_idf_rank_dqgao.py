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
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV


__author__ = ['chaonan99']


train_data = '../input/train.json'
test_data = '../input/test.json'


def run_pipeline(submission=False):
    np.random.seed(42)
    get_ingredients = lambda x: list(map(lambda x: x['ingredients'], x))
    get_cuisine = lambda x: list(map(lambda x: x['cuisine'], x))
    ingredient_format = lambda x: ' '.join(x)
    # ingredient_format = lambda x: ' '.join([i.replace(' ', '_') for i in x])

    train_content = json.load(open(train_data, encoding = 'utf-8'))
    test_content = json.load(open(test_data, encoding = 'utf-8'))
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

#    clf = svm.SVC(C=1000,
#                  kernel='rbf',
#                  # coef0=1,
#                  # class_weight=None,
#                  verbose=True)
#    clf = OneVsRestClassifier(clf)
#    clf.fit(X_tr, y_tr)

    text_clf = Pipeline([('clf', svm.SVC(C=100, # penalty parameter
                                         kernel='rbf', # kernel type, rbf working fine here
                                         degree=3, # default value
                                         gamma=1, # kernel coefficient
                                         coef0=1, # change to 1 from default value of 0.0
                                         shrinking=True, # using shrinking heuristics
                                         tol=0.001, # stopping criterion tolerance
                                         probability=False, # no need to enable probability estimates
                                         cache_size=200, # 200 MB cache size
                                         class_weight=None, # all classes are treated equally
                                         verbose=False, # print the logs
                                         max_iter=-1, # no limit, let it run
                                         decision_function_shape='ovr', # will use one vs rest explicitly
                                         random_state=None))
                        ])
    text_clf = text_clf.fit(X_tr, y_tr)

    y_dev_pred = text_clf.predict(X_dev)
    print("Accuracy:", accuracy_score(y_dev, y_dev_pred))

    parameters = {'clf__C': (1.0, 10, 100, 1000),
                  'clf__gamma': (0.001, 0.01, 0.1, 1)
                 }
    gs_clf = GridSearchCV(text_clf, parameters, cv=5, refit=True, n_jobs=1)
    gs_clf = gs_clf.fit(X_tr, y_tr)
    print(gs_clf.best_params_)

    if not submission:
        y_dev_pred = gs_clf.predict(X_dev)
        print("Accuracy:", accuracy_score(y_dev, y_dev_pred))
#    else:
#        y_te = clf.predict(X_te)
#        test_label_pred = le.inverse_transform(y_te)
#
#        test_id = list(map(lambda x: x['id'], test_content))
#        sub = pd.DataFrame({'id': test_id, 'cuisine': test_label_pred},
#                           columns=['id', 'cuisine'])
#        sub.to_csv('../dump/svm_output.csv', index=False)


def main():
    run_pipeline(submission=False)


if __name__ == '__main__':
    main()