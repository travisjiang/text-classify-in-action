from colorama import init
from termcolor import colored
import sklearn.datasets
import sklearn.feature_extraction.text
import sklearn.naive_bayes
import sklearn.cross_validation
import sklearn.svm
import sklearn.neighbors


def load_files(dataset_name):
    files = None
    if dataset_name == "20newsgroups":
        import data_util.newsgroups20 as newsgroups20
        #files = newsgroups20.load_files_2_categories("dataset/en/20newsgroups")
        files = newsgroups20.load_files_n_categories("dataset/en/20newsgroups")

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


def select_features(files, feature_type="bow"):
    if feature_type == "bow":
        return feature_bow(files.data)
    elif feature_type == "tfidf":
        return feature_tfidf(files.data)


def feature_bow(files_data):
    """
    Converts a list of strings (which are loaded from files) to a BOW representation of it
    parameter 'files_data' is a list of strings
    returns a `scipy.sparse.coo_matrix`
    """

    count_vector = sklearn.feature_extraction.text.CountVectorizer()
    return count_vector.fit_transform(files_data)


def feature_tfidf(files_data):
    word_counts = feature_bow(files_data)

    # TFIDF
    print colored('Calculating TFIDF', 'green', attrs=['bold'])
    tf_transformer = sklearn.feature_extraction.text.TfidfTransformer(
        use_idf=True).fit(word_counts)
    X = tf_transformer.transform(word_counts)
    return X


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


if __name__ == '__main__':
    main()
