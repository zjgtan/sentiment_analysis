# coding: utf8

import sys
reload(sys)
sys.setdefaultencoding("utf8")
from preprocess import load_dataset
import random
from gensim.models import doc2vec
from gensim.models.doc2vec import LabeledSentence
from sklearn.ensemble import RandomForestClassifier

import utils

import numpy as np

# Set values for various parameters
num_features = 300    # Word vector dimensionality
min_word_count = 40   # Minimum word count
num_workers = 4       # Number of threads to run in parallel
context = 10          # Context window size
downsampling = 1e-3   # Downsample setting for frequent words

def makeFeature(reviews, doc_emb):
    review_feature_vecs = []

    for review in reviews:
        feature_vec = doc_emb.infer_vector(review)
        review_feature_vecs.append(feature_vec)

    return review_feature_vecs


def main():
    train_reviews, train_labels, unlabeled_reviews, test_ids, test_reviews = load_dataset("../")

    print >> sys.stderr, "start training wordemb"

    documents = []
    for ix, review in enumerate(train_reviews):
        documents.append(LabeledSentence(review, ["TRAIN_" + str(ix)]))

    for ix, review in enumerate(unlabeled_reviews):
        documents.append(LabeledSentence(review, ["UNLABELED_" + str(ix)]))

    for ix, review in enumerate(test_reviews):
        documents.append(LabeledSentence(review, ["TEST_" + str(ix)]))

    model = doc2vec.Doc2Vec(vector_size=100, window=10, min_count=1, sample=1e-4, negative=5, workers=4)
    model.build_vocab(documents)

    model.train(documents, total_examples=len(documents), epochs=5)

    train_feature_vecs = []
    for ix in range(len(train_reviews)):
        train_feature_vecs.append(model["TRAIN_" + str(ix)])

    test_feature_vecs = []
    for ix in range(len(test_reviews)):
        test_feature_vecs.append(model["TEST_" + str(ix)])

    '''
    forest = RandomForestClassifier(n_estimators=100)
    forest = forest.fit(train_feature_vecs, train_labels)
    '''
    lr_model =

    # Test & extract results
    result = forest.predict(test_feature_vecs)

    # Write the test results
    utils.submit_result(test_ids, result, "../submission.csv")


if __name__ == "__main__":
    main()



