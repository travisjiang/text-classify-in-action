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
        "features": "chi_bow",
        "classifiers":"knn",
        "class_num":6
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
        "features": "chi_tfidf",
        "classifiers":"naive_bayes"
        }

config_5 = {
        "dataset": "sogou_cut",
        "features": "chi_bow",
        "classifiers":"naive_bayes",
        "cutted":True,
        "class_num":10
        }

config_6 = {
        "dataset": "netease_cut",
        "features": "chi_bow",
        "classifiers":"svm",
        "cutted":True,
        "class_num":6
        }

config = config_5



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
    dataset_name = config["dataset"]
    files = util.load_files(dataset_name)

    X = util.select_features(files, feature_type=config["features"],
            dataset_name = dataset_name, cutted=config.get("cutted", False))
    Y = files.target


    import numpy as np
    import lda
    import lda.datasets
    model = lda.LDA(n_topics=config["class_num"], n_iter=1500, random_state=1)
    model.fit(X)  # model.fit_transform(X) is also available
    topic_word = model.topic_word_  # model.components_ also works

    doc_topic = model.doc_topic_
    #for i in range(50):
    #    print("{} (top topic: {})".format(Y[i], doc_topic[i].argmax()))
    y_topic = [t.argmax() for t in doc_topic]
    import pdb
    pdb.set_trace()
    print sklearn.metrics.confusion_matrix(Y, y_topic)

#    import pdb
#    pdb.set_trace()

    #clf = util.select_classifiers(config["classifiers"])

    ## test the classifier
    #print '\n\n'
    #print colored('Testing classifier with train-test split', 'magenta', attrs=['bold'])
    #test_classifier(X, Y, clf, test_size=0.2, y_names=files.target_names, confusion=False)



if __name__ == '__main__':
    main()
