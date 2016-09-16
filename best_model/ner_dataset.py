#!/usr/bin/env python

import csv
import collections
import itertools
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle

from keras.utils import np_utils


def read_dataset(file_path):
    with open(file_path, 'r') as csvfile:
        ner_tags = csv.reader(csvfile, delimiter='\t')
        sent_words = []
        sent_tags = []

        words = []
        tags = []
        for row in ner_tags:
            if (len(row) > 0):
                words.append(row[0])
                tags.append(row[1])
            else:
                sent_words.append(words)
                sent_tags.append(tags)
                words = []
                tags = []

        if len(words) > 0:
            sent_words.append(words)
            sent_tags.append(tags)

    return sent_words, sent_tags


def build_embedding(file_path):
    with open(file_path, 'r') as csvfile:
        word_vectors = csv.reader(csvfile, delimiter='\t')
        word_embedding = {}
        embedding_vector_len = 0
        for a_word_vector in word_vectors:
            word = a_word_vector[0]
            vector = [1]    # a flag for seen words
            for i in range(1, len(a_word_vector)):
                if len(a_word_vector[i]) > 0:
                    vector.append(float(a_word_vector[i]))

            word_embedding[word] = vector
            embedding_vector_len = len(vector)

        word_embedding['UNK'] = [0 for i in range(embedding_vector_len)] # the first dimention indicates it is unknown word
    return word_embedding

def build_category(tags):
    counts = collections.Counter(itertools.chain(*tags))
    dictionary = dict()
    for tag in counts:
        dictionary[tag] = len(dictionary) + 1   # +1, 0 is reserved for padding
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))

    Y_train = list()
    for a_seq in tags:
        y_train = []
        for tag in a_seq:
            y_train.append(dictionary[tag])
        Y_train.append(y_train)

    return Y_train, dictionary, reverse_dictionary

def build_dataset(sents, embedding_words):
    word_counts = [['UNK', -1]]
    words = [normalize_word(word) for word in itertools.chain(*sents)]
    word_counts.extend(collections.Counter(words).items())

    dictionary = dict()
    for word, _ in word_counts:
        if word in embedding_words:
            dictionary[word] = len(dictionary) + 1 #zero will be used for padding

    X_word_train = list()
    X_char_train = list()
    unk_count = 0
    max_char = 0
    for sent in sents:
        x_word = []
        x_char = []
        for word in sent:
            chars = [ord(c) for c in normalize_word(word)]
            x_char.append(chars)
            if word in dictionary:
                index = dictionary[word]
            else:
                index = 1  # dictionary['UNK']
                unk_count = unk_count + 1
            x_word.append(index)
        X_word_train.append(x_word)
        X_char_train.append(x_char)
    word_counts[0][1] = unk_count
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return X_word_train, X_char_train, dictionary, reverse_dictionary

def convert_char_to_norm(c):
    if c.isdigit():
        return '0'
    if ord(c) < 128:
        return c.lower()
    return chr(128)

def normalize_word(word):
    return ''.join([convert_char_to_norm(c) for c in word])

def get_max(sents):
    sents_len = [len(sent) for sent in sents]
    words_len = [len(word) for sent in sents for word in sent]
    return max(sents_len), max(words_len)

def pad_dataset(X_word_train, X_char_train, Y_train, max_sent_len, max_word_len, nb_classes):
    print(nb_classes)
    if len(X_word_train) != len(X_char_train) or len(X_char_train) != len(Y_train):
        print('ERROR, words and chars have different length')

    X_word_train = pad_sequences(X_word_train, max_sent_len)
    Y_train = pad_sequences(Y_train, max_sent_len)

    X_char = []
    Y_one_hot = np.zeros((len(Y_train), max_sent_len, 10))
    for i in range(len(X_char_train)):
        padded_sequence = pad_sequences([[] for k in range(max_sent_len - len(X_char_train[i]))],
                                        maxlen=max_word_len).tolist() + \
                          pad_sequences(X_char_train[i], maxlen=max_word_len).tolist()
        X_char.append(padded_sequence)
        Y_one_hot[i] = np_utils.to_categorical(Y_train[i], nb_classes=nb_classes)

    return X_word_train, np.array(X_char), Y_one_hot

def main():
    sents, sents_tags = read_dataset('data/train.txt')
    word_embedding = build_embedding('data/wordvecs.txt')

    Y_train, tag_dict, tag_reverse_dict = build_category(sents_tags)
    X_word_train, X_char_train, dictionary, reverse_dictionary = build_dataset(sents, word_embedding)

    vocab_dim = 301 # dimensionality of your word vectors
    n_symbols = len(dictionary) + 1 # adding 1 to account for 0th index (for masking)
    embedding_weights = np.zeros((n_symbols,vocab_dim))
    for word,index in dictionary.items():
        embedding_weights[index,:] = word_embedding[word]

    print(sents[0])
    print(X_word_train[0])
    print(X_char_train[0])
    print(sents_tags[0])
    print(Y_train[0])

    max_sent_len, max_word_len = get_max(sents)
    print('Max Sent Len = {}, Max Word Len = {}'.format(max_sent_len, max_word_len))

    nb_classes = len(tag_dict) + 1 # +1 for padding
    X_word_train, X_char_train, Y_train = pad_dataset(X_word_train, X_char_train, Y_train, max_sent_len, max_word_len, nb_classes)

    print('Train_words={}, Train_chars={}, Tags={}'.format(X_word_train.shape, X_char_train.shape, Y_train.shape))
    train_dataset = {'word': X_word_train, 'char': X_char_train, 'tags': Y_train, 'embedding': embedding_weights,
                     'max sent len':max_sent_len, 'max word len':max_word_len, 'dictionary': dictionary,
                     'tag dictionary': tag_dict}

    with open('data/training.pkl', 'wb') as f:
        pickle.dump(train_dataset, f, pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    # execute only if run as a script
    main()
