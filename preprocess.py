# coding: utf8
import sys
import re
from nltk.corpus import stopwords
import pandas as pd
from bs4 import BeautifulSoup


def review_to_words(raw_review):
    review_text = BeautifulSoup(raw_review).get_text()
    letters_only = re.sub("[^a-zA-Z]", " ", review_text)
    words = letters_only.lower().split()
    stops = set(stopwords.words("english"))
    meaningful_words = [w for w in words if not w in stops]
    return meaningful_words


def do_review_clean(raw_review):
    return review_to_words(raw_review)

def load_labeled_train_data(filename):
    """加载训练样本
    :param filename:
    :return:
    """
    labels = []
    reviews = []
    for ix, line in enumerate(file(filename)):
        if ix == 0:
            continue
        toks = line.rstrip().split("\t")
        labels.append(int(toks[1]))
        reviews.append(do_review_clean(toks[2]))
    return reviews, labels


def load_unlabeled_train_data(filename):
    reviews = []
    for ix, line in enumerate(file(filename)):
        if ix == 0:
            continue
        toks = line.rstrip().split("\t")
        reviews.append(do_review_clean(toks[1]))
    return reviews


def load_test_data(filename):
    reviews = []
    ids = []
    for ix, line in enumerate(file(filename)):
        if ix == 0:
            continue
        toks = line.rstrip().split("\t")
        ids.append(toks[0])
        reviews.append(review_to_words(toks[1]))

    return ids, reviews


def load_dataset(data_dir):
    file_labeled_train_data = data_dir + "/labeledTrainData.tsv"
    file_unlabeled_train_data = data_dir + "/unlabeledTrainData.tsv"
    file_test_data = data_dir + "/testData.tsv"

    print >> sys.stderr, "load train data"
    train_reviews, train_labels = load_labeled_train_data(file_labeled_train_data)
    print >> sys.stderr, "load unlabeled data"
    unlabeled_reviews = load_unlabeled_train_data(file_unlabeled_train_data)
    print >> sys.stderr, "load test data"
    test_ids, test_reviews = load_test_data(file_test_data)

    return train_reviews, train_labels, unlabeled_reviews, test_ids, test_reviews

if __name__ == "__main__":
    load_dataset("../")

