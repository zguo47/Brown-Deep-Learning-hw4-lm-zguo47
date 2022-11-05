import tensorflow as tf
import numpy as np
from functools import reduce


def get_data(train_file, test_file):
    """
    Read and parse the train and test file line by line, then tokenize the sentences to build the train and test data separately.
    Create a vocabulary dictionary that maps all the unique tokens from your train and test data as keys to a unique integer value.
    Then vectorize your train and test data based on your vocabulary dictionary.

    :param train_file: Path to the training file.
    :param test_file: Path to the test file.
    :return: Tuple of:
        train (1-d list or array with training words in vectorized/id form), 
        test (1-d list or array with testing words in vectorized/id form), 
        vocabulary (Dict containg word->index mapping)
    """
    vocabulary, vocab_size, train_data, test_data = {}, 0, [], []

    ## TODO: Implement pre-processing for the data files. See notebook for help on this.

    with open(train_file) as f:
        word_list_train = []
        unique_word_list_train = []
        line_train = f.readlines()
        for l in line_train:
            word_list_train.extend(l.lower().split())

    
    with open(test_file) as f:
        word_list_test = []
        unique_word_list_test = []
        line_test = f.readlines()
        for l in line_test:
            word_list_test.extend(l.lower().split())


    index = len(word_list_train)
    word_list_train.extend(word_list_test)
    unique_word_list = sorted(set(word_list_train))
    vocabulary = {w: i for i, w in enumerate(unique_word_list)}
    # train_data = [vocabulary[w] for w in word_list_train[0:index]]
    # test_data = [vocabulary[w] for w in word_list_train[index:]]
    train_data = word_list_train[0:index]
    test_data = word_list_train[index:]
    vocab_size = len(vocabulary.keys())
    
    # Sanity Check, make sure there are no new words in the test data.
    assert reduce(lambda x, y: x and (y in vocabulary), test_data)

    # Vectorize, and return output tuple.
    train_data = list(map(lambda x: vocabulary[x], train_data))
    test_data  = list(map(lambda x: vocabulary[x], test_data))

    # print(train_data)
    # print(test_data)
    # print(vocabulary)

    # print("train_data", train_data)
    return train_data, test_data, vocabulary
