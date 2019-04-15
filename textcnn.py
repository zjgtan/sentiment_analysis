# coding: utf8

import sys
reload(sys)
sys.setdefaultencoding("utf8")
from preprocess import load_dataset
from gensim.models import doc2vec
from gensim.models.doc2vec import TaggedDocument

import tensorflow as tf

import utils

class TextCNN:


def main():
    train_reviews, train_labels, unlabeled_reviews, test_ids, test_reviews = load_dataset("../")

    # 构建语料集
    documents = []
    for ix, review in enumerate(train_reviews):
        documents.append(TaggedDocument(review, [ix]))

    for ix, review in enumerate(unlabeled_reviews):
        documents.append(TaggedDocument(review, [ix + len(train_reviews)]))

    doc_emb = doc2vec.Doc2Vec(documents, workers=num_workers,
                             vector_size=num_features, min_count=min_word_count,
                             window=context)


    train_feature_vecs = makeFeature(train_reviews, doc_emb)
    test_feature_vecs = makeFeature(test_reviews, doc_emb)

    forest = RandomForestClassifier(n_estimators=100)
    forest = forest.fit(train_feature_vecs, train_labels)

    # Test & extract results
    result = forest.predict(test_feature_vecs)

    # Write the test results
    utils.submit_result(test_ids, result, "../submission.csv")


if __name__ == "__main__":
    main()



