import sys
import collections
import sklearn.naive_bayes
import sklearn.linear_model
import nltk
import random

random.seed(0)
from gensim.models.doc2vec import LabeledSentence, Doc2Vec

nltk.download("stopwords")  # Download the stop words from nltk

# LabeledSentence = gensim.models.doc2vec.LabeledSentence # I added this piece of code


# User input path to the train-pos.txt, train-neg.txt, test-pos.txt, and test-neg.txt datasets
if len(sys.argv) != 3:
    print "python sentiment.py <path_to_data> <0|1>"
    print "0 = NLP, 1 = Doc2Vec"
    exit(1)
path_to_data = sys.argv[1]
method = int(sys.argv[2])


def main():
    train_pos, train_neg, test_pos, test_neg = load_data(path_to_data)

    if method == 0:
        train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec = feature_vecs_NLP(train_pos, train_neg, test_pos,
                                                                                    test_neg)
        nb_model, lr_model = build_models_NLP(train_pos_vec, train_neg_vec)
    if method == 1:
        train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec = feature_vecs_DOC(train_pos, train_neg, test_pos,
                                                                                    test_neg)
        nb_model, lr_model = build_models_DOC(train_pos_vec, train_neg_vec)
    print "Naive Bayes"
    print "-----------"
    evaluate_model(nb_model, test_pos_vec, test_neg_vec, True)
    print ""
    print "Logistic Regression"
    print "-------------------"
    evaluate_model(lr_model, test_pos_vec, test_neg_vec, True)


def load_data(path_to_dir):
    """
    Loads the train and test set into four different lists.
    """
    train_pos = []
    train_neg = []
    test_pos = []
    test_neg = []
    with open(path_to_dir + "train-pos.txt", "r") as f:
        for i, line in enumerate(f):
            words = [w.lower() for w in line.strip().split() if len(w) >= 3]
            train_pos.append(words)
    with open(path_to_dir + "train-neg.txt", "r") as f:
        for line in f:
            words = [w.lower() for w in line.strip().split() if len(w) >= 3]
            train_neg.append(words)
    with open(path_to_dir + "test-pos.txt", "r") as f:
        for line in f:
            words = [w.lower() for w in line.strip().split() if len(w) >= 3]
            test_pos.append(words)
    with open(path_to_dir + "test-neg.txt", "r") as f:
        for line in f:
            words = [w.lower() for w in line.strip().split() if len(w) >= 3]
            test_neg.append(words)

    return train_pos, train_neg, test_pos, test_neg


def createDict(dataset, stopwords):
    tmpDict = {}
    for wordList in dataset:
        for word in wordList:
            if word not in stopwords:  # Condition 1
                if word not in tmpDict:
                    tmpDict[word] = 0
                tmpDict[word] += 1
    return tmpDict


def createFeatureList(dataset, trainDict1, trainDict2):
    tmpDataset = []
    for word in trainDict1:
        if trainDict1[word] >= (0.01) * len(dataset):  # Condition 2
            if trainDict1[word] >= 2 * trainDict2.get(word, 0):  # [word]: #Condition 3
                tmpDataset.append(word)
    return tmpDataset


def createVec(dataset, features):
    tmp_vec = []
    for wordList in dataset:
        tmp_list = []
        for aFeature in features:
            if aFeature in wordList:
                tmp_list.append(1)
            else:
                tmp_list.append(0)
        tmp_vec.append(tmp_list)
    return tmp_vec


def feature_vecs_NLP(train_pos, train_neg, test_pos, test_neg):
    """
    Returns the feature vectors for all text in the train and test datasets.
    """
    # English stopwords from nltk
    stopwords = set(nltk.corpus.stopwords.words('english'))

    # Determine a list of words that will be used as features.
    # This list should have the following properties:
    #   (1) Contains no stop words
    #   (2) Is in at least 1% of the positive texts or 1% of the negative texts
    #   (3) Is in at least twice as many postive texts as negative texts, or vice-versa.
    # YOUR CODE HERE

    train_pos_new = []
    train_neg_new = []
    trainPosDict = {}
    trainNegDict = {}

    # Counting frequency of each words in TRAIN_POS into a dictionary
    trainPosDict = createDict(train_pos, stopwords)
    # Counting frequency of each words in TRAIN_NEG into a dictionary
    trainNegDict = createDict(train_neg, stopwords)

    # Creating list with the given conditions - for train_pos and train_neg - (Note the arguments of the function)
    train_pos_new = createFeatureList(train_pos, trainPosDict, trainNegDict)
    train_neg_new = createFeatureList(train_neg, trainNegDict, trainPosDict)

    features = train_pos_new + train_neg_new
    features = list(set(features))

    # Using the above words as features, construct binary vectors for each text in the training and test set.
    # These should be python lists containing 0 and 1 integers.
    # YOUR CODE HERE
    train_pos_vec = []
    train_neg_vec = []
    test_pos_vec = []
    test_neg_vec = []

    train_pos_vec = createVec(train_pos, features)
    train_neg_vec = createVec(train_neg, features)
    test_pos_vec = createVec(test_pos, features)
    test_neg_vec = createVec(test_neg, features)

    # Return the four feature vectors
    return train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec


def createLabel(dataset, label):
    tmpList = []
    for i in range(0, len(dataset)):
        tmpList.append(LabeledSentence(dataset[i], [label + str(i)]))
    return tmpList


def feature_vecs_DOC(train_pos, train_neg, test_pos, test_neg):
    """
    Returns the feature vectors for all text in the train and test datasets.
    """
    # Doc2Vec requires LabeledSentence objects as input.
    # Turn the datasets from lists of words to lists of LabeledSentence objects.
    # YOUR CODE HERE
    labeled_train_pos = []
    labeled_train_neg = []
    labeled_test_pos = []
    labeled_test_neg = []

    labeled_train_pos = createLabel(train_pos, 'TRAIN_POS_')
    labeled_train_neg = createLabel(train_neg, 'TRAIN_NEG_')
    labeled_test_pos = createLabel(test_pos, 'TEST_POS_')
    labeled_test_neg = createLabel(test_neg, 'TEST_NEG_')

    # Initialize model
    model = Doc2Vec(min_count=1, window=10, size=100, sample=1e-4, negative=5, workers=4)
    sentences = labeled_train_pos + labeled_train_neg + labeled_test_pos + labeled_test_neg
    model.build_vocab(sentences)

    # Train the model
    # This may take a bit to run
    for i in range(5):
        print "Training iteration %d" % (i)
        random.shuffle(sentences)
        model.train(sentences)

    # Use the docvecs function to extract the feature vectors for the training and test data
    # YOUR CODE HERE
    train_pos_vec = [model.docvecs['TRAIN_POS_' + str(i)] for i in range(0, len(train_pos))]
    train_neg_vec = [model.docvecs['TRAIN_NEG_' + str(i)] for i in range(0, len(train_neg))]
    test_pos_vec = [model.docvecs['TEST_POS_' + str(i)] for i in range(0, len(test_pos))]
    test_neg_vec = [model.docvecs['TEST_NEG_' + str(i)] for i in range(0, len(test_neg))]

    # Return the four feature vectors
    return train_pos_vec, train_neg_vec, test_pos_vec, test_neg_vec


def build_models_NLP(train_pos_vec, train_neg_vec):
    """
    Returns a BernoulliNB and LosticRegression Model that are fit to the training data.
    """
    Y = ["pos"] * len(train_pos_vec) + ["neg"] * len(train_neg_vec)

    # Use sklearn's BernoulliNB and LogisticRegression functions to fit two models to the training data.
    # For BernoulliNB, use alpha=1.0 and binarize=None
    # For LogisticRegression, pass no parameters
    # YOUR CODE HERE

    trainData = train_pos_vec + train_neg_vec
    nb_model = sklearn.naive_bayes.BernoulliNB(alpha=1.0, binarize=None)
    nb_model.fit(trainData, Y)

    lr_model = sklearn.linear_model.LogisticRegression()
    lr_model.fit(trainData, Y)

    return nb_model, lr_model


def build_models_DOC(train_pos_vec, train_neg_vec):
    """
    Returns a GaussianNB and LosticRegression Model that are fit to the training data.
    """
    Y = ["pos"] * len(train_pos_vec) + ["neg"] * len(train_neg_vec)

    # Use sklearn's GaussianNB and LogisticRegression functions to fit two models to the training data.
    # For LogisticRegression, pass no parameters
    # YOUR CODE HERE
    trainData = train_pos_vec + train_neg_vec

    nb_model = sklearn.naive_bayes.GaussianNB()
    nb_model.fit(trainData, Y)

    lr_model = sklearn.linear_model.LogisticRegression()
    lr_model.fit(trainData, Y)

    return nb_model, lr_model


def evaluate_model(model, test_pos_vec, test_neg_vec, print_confusion=False):
    """
    Prints the confusion matrix and accuracy of the model.
    """
    # Use the predict function and calculate the true/false positives and true/false negative.
    # YOUR CODE HERE

    pos_predict = model.predict(test_pos_vec)
    neg_predict = model.predict(test_neg_vec)

    tp = 0
    fn = 0
    fp = 0
    tn = 0

    for prediction in pos_predict:
        if prediction == "pos":
            tp += 1
        else:
            fn += 1

    for prediction in neg_predict:
        if prediction == "neg":
            tn += 1
        else:
            fp += 1

    accuracy = float(tp + tn) / float(tp + tn + fp + fn)

    if print_confusion:
        print "predicted:\tpos\tneg"
        print "actual:"
        print "pos\t\t%d\t%d" % (tp, fn)
        print "neg\t\t%d\t%d" % (fp, tn)
    print "accuracy: %f" % (accuracy)


if __name__ == "__main__":
    main()