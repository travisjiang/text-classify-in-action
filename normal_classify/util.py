# -*- coding: UTF-8 -*-

from colorama import init
from termcolor import colored
import sklearn.datasets
import sklearn.feature_extraction.text
import sklearn.naive_bayes
import sklearn.cross_validation
import sklearn.svm
import sklearn.neighbors

import os
import nltk.tokenize
#from nltk.tokenize import WordPunctTokenizer

dataset_path_map = {
        "20newsgroups": "dataset/en/20newsgroups",
        "sogou_cut": "dataset/cn/SogouCCut",
        "netease_cut" : "dataset/cn/netease"
        }


def load_files(dataset_name):
    files = None
    path = dataset_path_map[dataset_name]
    if dataset_name == "20newsgroups":
        import data_util.newsgroups20 as newsgroups20
        #files = newsgroups20.load_files_2_categories(path)
        files = newsgroups20.load_files_n_categories(path)
    if dataset_name == "sogou_cut":
        import data_util.sogou as sogou
        files = sogou.load_files_n_categories(path)
    if dataset_name == "netease_cut":
        import data_util.netease as netease
        files = netease.load_files_n_categories(path)

    return files



def select_classifiers(clf_name="knn"):
    clf = None
    if clf_name == "knn":
        n_neighbors = 11
        weights = 'uniform'
        weights = 'distance'
        clf = sklearn.neighbors.KNeighborsClassifier(
            n_neighbors, weights=weights)
    elif clf_name == "naive_bayes":
        clf = sklearn.naive_bayes.MultinomialNB()
    elif clf_name == "svm":
        clf = sklearn.svm.LinearSVC()

    return clf


def select_features(files, feature_type, dataset_name, cutted=False):
    if feature_type == "bow":
        return feature_bow(files.data)
    elif feature_type == "tfidf":
        return feature_tfidf(files.data)
    elif feature_type == "chi_tfidf":
        return feature_chi(files, dataset_name, cutted=cutted, weight="tfidf")
    elif feature_type == "chi_bow":
        return feature_chi(files, dataset_name, cutted=cutted, weight="bow")


def feature_bow(files_data):
    """
    Converts a list of strings (which are loaded from files) to a BOW representation of it
    parameter 'files_data' is a list of strings
    returns a `scipy.sparse.coo_matrix`
    """

    count_vector = sklearn.feature_extraction.text.CountVectorizer(decode_error='ignore')
    return count_vector.fit_transform(files_data)


def feature_tfidf(files_data):
    word_counts = feature_bow(files_data)

    # TFIDF
    print colored('Calculating TFIDF', 'green', attrs=['bold'])
    tf_transformer = sklearn.feature_extraction.text.TfidfTransformer(
        use_idf=True).fit(word_counts)
    X = tf_transformer.transform(word_counts)
    return X


def word_tokenize(doc, space_seg=False):
    #words_set = nltk.tokenize.WordPunctTokenizer().tokenize(doc)
    if space_seg:
        import re
        r = re.compile(r"\s+")
        return r.split(doc)
    else:
        return nltk.tokenize.TweetTokenizer().tokenize(doc)


# 对卡方检验所需的 a b c d 进行计算
# a：在这个分类下包含这个词的文档数量
# b：不在该分类下包含这个词的文档数量
# c：在这个分类下不包含这个词的文档数量
# d：不在该分类下，且不包含这个词的文档数量

def calc_chi(a, b, c, d):
    result = float(pow(a*d-b*c, 2)) / float((a+c)*(a+b)*(b+d)*(c+d))
    return result

def feature_chi(files, dataset_name, update=False, cutted = False, weight="tfidf"):
    print("select feature: chi square")

    feature_chi_path = dataset_path_map[dataset_name]+"/feature_chi.txt"

    if not os.path.exists(feature_chi_path) or update:
        print("generate chi feature file...")

        total_doc_num, df_in_class_term, df_in_class, df_in_term = calc_chi_var(files, cutted)

        chi_term_set = get_chi_term(df_in_class_term, df_in_class, df_in_term, total_doc_num, 1000)

        save2file_feature(feature_chi_path, chi_term_set)

    else:
        print("loading chi feature file...")

        chi_term_set = load_feature(feature_chi_path)


    print("convert data using chi feature...")
    chi_data = convert_data_with_chi(files.data, chi_term_set, cutted)

    print("calc chi feature weights by %s..." % weight)
    if weight == "bow":
        return feature_bow(chi_data)
    return feature_tfidf(chi_data)

def convert_data_with_chi(data, chi_term_set, cutted):
    new_data = []
    for i, doc in enumerate(data):
        new_doc = [w for w in word_tokenize(doc, cutted) if w in chi_term_set]
        new_data.append(" ".join(new_doc))
    return new_data



def calc_chi_var(files, cutted):
    total_doc_num = 0;

    #doc frequency: in a class, and contains term
    df_in_class_term = {}

    #doc frequency: contains term
    df_in_term = {}

    #doc_frequency: in a class
    df_in_class = {cls:0 for cls in range(len(files.target_names))}

    for i, doc in enumerate(files.data):
        doc_class = files.target[i]

        df_in_class[doc_class] += 1

        if not df_in_class_term.get(doc_class, None):
            df_in_class_term[doc_class] = {}
        df_tmp = df_in_class_term[doc_class]

        total_doc_num += 1

        words_set = word_tokenize(doc, cutted)
        for word in words_set:
            if not df_tmp.get(word, None):
                df_tmp[word] = 1
            else:
                df_tmp[word] += 1

            if not df_in_term.get(word, None):
                df_in_term[word] = 1
            else:
                df_in_term[word] += 1

    return total_doc_num, df_in_class_term, df_in_class, df_in_term


def get_chi_term(df_in_class_term, df_in_class, df_in_term, total_doc_num, k):
    chi_term_set = set()
    for cls in df_in_class_term.keys():
        df_tmp = df_in_class_term[cls]
        term_chi = {}
        for word in df_tmp:
            a = df_tmp[word]
            b = df_in_term[word]-a
            doc_not_contain_term = total_doc_num - df_in_term[word]
            c = df_in_class[cls] - a
            d = doc_not_contain_term - c
            term_chi[word] = calc_chi(a,b,c,d)

        # this will return a sorted list of tuple
        sorted_term_chi = sorted(term_chi.items(), key=lambda d:d[1], reverse=True)

        l = k if k < len(sorted_term_chi) else len(sorted_term_chi)
        for i in range(l):
            chi_term_set.add(sorted_term_chi[i][0])
        #import pdb
        #pdb.set_trace()

    return chi_term_set

def save2file_feature(path, feature_set):

    with open(path, 'w') as f:
        for feature in feature_set:
            f.write(feature + "\n")

def load_feature(path):
    feature_set = None
    with open(path, 'r') as f:
        feature_set = f.readlines()
        feature_set = [str.strip(l) for l in feature_set if str.strip(l) != '']

    return set(feature_set)








def main():
    from colorama import init
    from termcolor import colored
    init()

    test_main()


def test_main():
    directory = 'ds2'
    directory = 'dataset'
    directory = 'ds3'
    # load the dataset from disk
    files = sklearn.datasets.load_files(directory)

    # refine them
    refine_all_emails(files.data)

    # calculate the BOW representation
    word_counts = bagOfWords(files.data)

    # TFIDF
    tf_transformer = sklearn.feature_extraction.text.TfidfTransformer(
        use_idf=True).fit(word_counts)
    X_tfidf = tf_transformer.transform(word_counts)

    X = X_tfidf

    # cross validation
    # clf = sklearn.naive_bayes.MultinomialNB()
    # clf = sklearn.svm.LinearSVC()
    n_neighbors = 5
    weights = 'uniform'
    # weights = 'distance'
    clf = sklearn.neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
    scores = cross_validation(X, files.target, clf, cv=5)
    pretty_print_scores(scores)


def pretty_print_scores(scores):
    """
    Prints mean and std of a list of scores, pretty and colorful!
    parameter `scores` is a list of numbers.
    """
    print colored("                                      ", 'white', 'on_white')
    print colored(" Mean accuracy: %0.3f (+/- %0.3f std) " % (scores.mean(), scores.std() / 2), 'magenta', 'on_white', attrs=['bold'])
    print colored("                                      ", 'white', 'on_white')


def cross_validation(data, target, classifier, cv=5):
    """
    Does a cross validation with the classifier
    parameters:
        - `data`: array-like, shape=[n_samples, n_features]
            Training vectors
        - `target`: array-like, shape=[n_samples]
            Target values for corresponding training vectors
        - `classifier`: A classifier from the scikit-learn family would work!
        - `cv`: number of times to do the cross validation. (default=5)
    return a list of numbers, where the length of the list is equal to `cv` argument.
    """
    return sklearn.cross_validation.cross_val_score(classifier, data, target, cv=cv)

def test_chi_stopwords(chi_path, stop_path):
    chi_set, stop_set = None, None
    with open(chi_path, 'r') as f:
        lines = f.readlines()
        words_list = [w.strip() for w in lines]
        chi_set = set(words_list)

    with open(stop_path, 'r') as f:
        lines = f.readlines()
        words_list = [w.strip() for w in lines]
        stop_set = set(words_list)

    return chi_set & stop_set


if __name__ == '__main__':
    #main()
    #####################test stop_words and chisquare term########################
    #same_part = test_chi_stopwords("./dataset/cn/SogouCCut/feature_chi.txt", "./dataset/hanlp_stopwords.txt")
    same_part = test_chi_stopwords("./dataset/cn/netease/feature_chi.txt", "./dataset/hanlp_stopwords.txt")
    print(len(same_part))
    print(same_part)
