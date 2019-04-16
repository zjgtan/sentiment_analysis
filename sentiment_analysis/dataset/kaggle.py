# coding: utf8
import re
from bs4 import BeautifulSoup
from nltk.corpus import stopwords

class KaggleDataset(object):
    def __init__(self, data_dir):
        file_labeled_train_data = data_dir + "/labeledTrainData.tsv"
        file_unlabeled_train_data = data_dir + "/unlabeledTrainData.tsv"
        file_test_data = data_dir + "/testData.tsv"

        self.dataset = {}
        train_reviews, train_labels, word_dict = self.load_labeled_train_data(file_labeled_train_data)
        #unlabeled_reviews = self.load_unlabeled_train_data(file_unlabeled_train_data)
        test_ids, test_reviews = self.load_test_data(file_test_data)

        self.dataset["train"] = {}
        self.dataset["train"]["reviews"] = train_reviews
        self.dataset["train"]["labels"] = train_labels

        self.dataset["word_dict"] = word_dict

        #self.dataset["unlabeled"] = {}
        #self.dataset["unlabeled"]["reviews"] = unlabeled_reviews

        self.dataset["test"] = {}
        self.dataset["test"]["reviews"] = test_reviews
        self.dataset["test"]["ids"] = test_ids

    def review_to_words(self, raw_review):
        review_text = BeautifulSoup(raw_review).get_text()
        letters_only = re.sub("[^a-zA-Z]", " ", review_text)
        words = letters_only.lower().split()
        stops = set(stopwords.words("english"))
        meaningful_words = [w for w in words if not w in stops]
        return meaningful_words

    def do_review_clean(self, raw_review):
        return self.review_to_words(raw_review)

    def load_labeled_train_data(self, filename):
        """加载训练样本
        """
        labels = []
        reviews = []
        word_dict = {}
        for ix, line in enumerate(file(filename)):
            if ix == 0:
                continue
            toks = line.rstrip().split("\t")
            labels.append(int(toks[1]))
            review = self.do_review_clean(toks[2])
            reviews.append(review)

            for word in review:
                if word not in word_dict:
                    word_dict[word] = len(word_dict)

        return reviews, labels, word_dict

    def load_unlabeled_train_data(self, filename):
        reviews = []
        for ix, line in enumerate(file(filename)):
            if ix == 0:
                continue
            toks = line.rstrip().split("\t")
            reviews.append(self.do_review_clean(toks[1]))
        return reviews

    def load_test_data(self, filename):
        reviews = []
        ids = []
        for ix, line in enumerate(file(filename)):
            if ix == 0:
                continue
            toks = line.rstrip().split("\t")
            ids.append(toks[0])
            reviews.append(self.review_to_words(toks[1]))

        return ids, reviews


if __name__ == "__main__":
    obj = KaggleDataset("../../kaggle_data")

