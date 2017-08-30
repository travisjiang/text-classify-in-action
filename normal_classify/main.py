import sklearn.datasets
import sklearn.metrics
import sklearn.cross_validation
import sklearn.svm
import sklearn.naive_bayes
import sklearn.neighbors
from colorama import init
from termcolor import colored
import sys
import os

import util

init()

config_1 = {
        "dataset": "20newsgroups",
        "features": "tfidf",
        "classifiers":"knn"
        }

config_2 = {
        "dataset": "20newsgroups",
        "features": "chi_tfidf",
        "classifiers":"knn"
        }

config_3 = {
        "dataset": "20newsgroups",
        "features": "tfidf",
        "classifiers":"svm"
        }

config_4 = {
        "dataset": "20newsgroups",
        "features": "bow",
        "classifiers":"naive_bayes"
        }

config = config_4



def test_classifier(X, y, clf, test_size=0.4, y_names=None, confusion=False):
    # train-test split
    print 'test size is: %2.0f%%' % (test_size * 100)
    X_train, X_test, y_train, y_test = sklearn.cross_validation.train_test_split(X, y, test_size=test_size)

    clf.fit(X_train, y_train)
    y_predicted = clf.predict(X_test)

    if not confusion:
        print colored('Classification report:', 'magenta', attrs=['bold'])
        print sklearn.metrics.classification_report(y_test, y_predicted, target_names=y_names)
    else:
        print colored('Confusion Matrix:', 'magenta', attrs=['bold'])
        print sklearn.metrics.confusion_matrix(y_test, y_predicted)


def main():
    # load_files
    files = util.load_files(config["dataset"])

    X = util.select_features(files, feature_type=config["features"])
    Y = files.target

#    import pdb
#    pdb.set_trace()

    clf = util.select_classifiers(config["classifiers"])

    # test the classifier
    print '\n\n'
    print colored('Testing classifier with train-test split', 'magenta', attrs=['bold'])
    test_classifier(X, Y, clf, test_size=0.2, y_names=files.target_names, confusion=False)



if __name__ == '__main__':
    main()
