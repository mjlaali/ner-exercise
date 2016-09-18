import collections
import csv
import itertools
import os

import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.utils import np_utils

from ner.category_manager import CategoryManager
from ner.char_vector_converter import CharVectorConverter
from ner.pretrained_word_vector_reader import PretrainedWordVectorReader


def convert_char_to_norm(c):
    if c.isdigit():
        return '0'
    if ord(c) < 128:
        return c
    return chr(128)

def normalize_word(word):
    max_word = 20
    if len(word) > max_word:
        # print(word)
        prefix = max_word // 2
        postfix = len(word) - max_word // 2
        word = word[:prefix] + word[postfix:]
        # print('\t' + word)

    return ''.join([convert_char_to_norm(c) for c in word])

def pad_sequences_2D(sequence, max_dim_1, max_dim_2, dtype='int32'):
    X = []
    for i in range(len(sequence)):
        padded_sequence = pad_sequences([[] for k in range(max_dim_1 - len(sequence[i]))],
                                        maxlen=max_dim_2, dtype=dtype, value=1).tolist() + \
                          pad_sequences(sequence[i], maxlen=max_dim_2, dtype=dtype, value=1).tolist()
        X.append(padded_sequence)
    return np.array(X)

class NnPreprocessor:

    def __init__(self, train_file, max_sent_len):
        self.sent_words, self.sent_tags = NnPreprocessor.read_tagged_file(train_file, max_sent_len)

        pass

    @staticmethod
    def read_tagged_file(file_path, max_sent_len):
        if max_sent_len == -1:
            max_sent_len = 40
        with open(file_path, 'r') as csvfile:
            ner_tags = csv.reader(csvfile, delimiter='\t', quoting=csv.QUOTE_NONE)
            sent_words = []
            sent_tags = []

            words = []
            tags = []
            for row in ner_tags:
                if (len(row) > 0):
                    words.append(normalize_word(row[0]))
                    tags.append(row[1])
                else:
                    if len(words) < max_sent_len:
                        sent_words.append(words)
                        sent_tags.append(tags)
                    else:
                        print('Warning an instance was removed from the dataset: {}'.format(' '.join(words)))
                    words = []
                    tags = []

            if len(words) > 0:
                sent_words.append(words)
                sent_tags.append(tags)

        return sent_words, sent_tags

    def __set_max_sent_and_word(self):
        sents_len = [len(sent) for sent in self.sent_words]
        words_len = [len(word) for sent in self.sent_words for word in sent]
        print('Sent len={}'.format(collections.Counter(sents_len).most_common()))
        print('Word len={}'.format(collections.Counter(words_len).most_common()))
        self.max_sent_len = max(sents_len)
        self.max_word_len = max(words_len)

    def get_train_dataset(self):
        return self.sent_words, self.sent_tags


def load(train_dataset_file, word_vector_file, delimiter='\t', max_sent_len=-1, max_word_len=-1, load_word_vectors=True):
    output = dict()

    preprocessor = NnPreprocessor(train_dataset_file, max_sent_len)
    sent_words, sent_tags = preprocessor.get_train_dataset()
    output['sent_words'] = sent_words
    output['sent_tags'] = sent_tags

    char_vector_converter = CharVectorConverter()
    sent_words_chars, dataset_max_sent_len, dataset_max_word_len = char_vector_converter.get_char_vectors(sent_words, max_word_len)
    output['sent_words_chars'] = sent_words_chars

    if max_sent_len == -1:
        max_sent_len = dataset_max_sent_len
    if max_word_len == -1:
        max_word_len = dataset_max_word_len
    output['max_sent_len'] = max_sent_len
    output['max_word_len'] = max_word_len

    print('Max sent len={}, Max word len={}'.format(max_sent_len, max_word_len))

    X_chars = pad_sequences_2D(sent_words_chars, max_sent_len, max_word_len)
    char_vocab_size = char_vector_converter.get_char_vocab_size()
    output['X_chars'] = X_chars
    output['char_vocab_size'] = char_vocab_size

    if load_word_vectors:
        reader = PretrainedWordVectorReader(word_vector_file)
        reader.load(set(itertools.chain(*sent_words)), delimiter=delimiter)
        sent_words_vecs = reader.convert_dataset(sent_words)
        word_embedding_size = reader.get_len_embedding_vector()
        X_words = pad_sequences_2D(sent_words_vecs, max_sent_len, word_embedding_size, dtype='float32')

        output['sent_words_vecs'] = sent_words_vecs
        output['word_embedding_size'] = word_embedding_size
        output['X_words'] = X_words

    category_manager = CategoryManager(sent_tags)
    Y = category_manager.pad_tags(sent_tags, max_sent_len)
    nb_classes = category_manager.get_num_classes()
    output['Y'] = Y
    output['nb_classes'] = nb_classes
    return output

def main():
    # train_dataset_file = '../data/coling2016/train'
    # word_vector_file = '../data/coling2016/glove.twitter.27B.200d.txt'
    train_dataset_file = os.path.join(os.path.dirname(__file__), '../data/maluuba/train.txt')
    word_vector_file = os.path.join(os.path.dirname(__file__), '../data/maluuba/wordvecs.txt')


    output = load(train_dataset_file, word_vector_file, delimiter='\t')

    print('sample :\nword = {} \ntags = {} \nchar = {} \n\tAfter padding:\nword = {} \ntags = {} \nchar = {}'.format(
        len(output['X_words']), len(output['Y']), len(output['X_chars']), output['X_words'].shape, output['Y'].shape, output['X_chars'].shape))

    for i in range(1):
        print(output['sent_words'][i])
        print(output['sent_tags'][i])
        print('sample {}:\nword = {} \ntags = {} \nchar = {} \n\tAfter padding:\nword = {} \ntags = {} \nchar = {}'.format(
            i, output['sent_words_vecs'][i], output['sent_tags'][i], output['sent_words_chars'][i],
            output['X_words'][i], output['Y'][i], output['X_chars'][i]))

if __name__ == "__main__":
    # execute only if run as a script
    main()