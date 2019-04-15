import sys
reload(sys)
sys.setdefaultencoding("utf8")
from preprocess import load_dataset
from gensim.models import word2vec
from sklearn.ensemble import RandomForestClassifier

import utils

import numpy as np

# Set values for various parameters
num_features = 300    # Word vector dimensionality
min_word_count = 40   # Minimum word count
num_workers = 4       # Number of threads to run in parallel
context = 10          # Context window size
downsampling = 1e-3   # Downsample setting for frequent words

def makeAvgFeature(reviews, word_emb, emb_dim):
    review_feature_vecs = []

    for review in reviews:
        feature_vec = np.zeros((emb_dim, ), dtype="float32")

        nwords = 0
        for word in review:
            if word in word_emb:
                nwords += 1
                feature_vec = np.add(feature_vec, word_emb[word])

        if nwords > 0:
            feature_vec = np.divide(feature_vec, nwords)

        review_feature_vecs.append(feature_vec)

    return review_feature_vecs


def main():
    train_reviews, train_labels, unlabeled_reviews, test_ids, test_reviews = load_dataset("../")

    print >> sys.stderr, "start training wordemb"
    word_emb = word2vec.Word2Vec(train_reviews + unlabeled_reviews, workers=num_workers,
                             size=num_features, min_count=min_word_count,
                             window=context, sample=downsampling)


    train_feature_vecs = makeAvgFeature(train_reviews, word_emb, num_features)
    test_feature_vecs = makeAvgFeature(test_reviews, word_emb, num_features)

    forest = RandomForestClassifier(n_estimators=100)
    forest = forest.fit(train_feature_vecs, train_labels)

    # Test & extract results
    result = forest.predict(test_feature_vecs)

    # Write the test results
    utils.submit_result(test_ids, result, "../submission.csv")


if __name__ == "__main__":
    main()



