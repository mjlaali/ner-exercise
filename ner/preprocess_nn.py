import csv
import collections
import itertools
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle
from keras.utils import np_utils

def convert_char_to_norm(c):
    if c.isdigit():
        return '0'
    if ord(c) < 128:
        return c.lower()
    return chr(128)

def normalize_word(word):
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

    def __init__(self, train_file, embedding_file):
        self.__read_embedding(embedding_file)
        self.sent_words, self.sent_tags = NnPreprocessor.read_tagged_file(train_file)
        self.__construct_word_dictionary()
        self.__construct_tag_dictionary()
        self.nb_classes = len(self.tag_dictionary) + 1  # +1: 0 is reserved for padding
        self.__set_max_sent_and_word()
        pass

    @staticmethod
    def read_tagged_file(file_path):
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

    def __read_embedding(self, file_path):
        with open(file_path, 'r') as csvfile:
            word_vectors = csv.reader(csvfile, delimiter='\t')
            word_embedding = {}
            embedding_vector_len = 0
            for a_word_vector in word_vectors:
                word = a_word_vector[0]
                vector = [1]    # first cell is a flag for seen words
                for i in range(1, len(a_word_vector)):
                    if len(a_word_vector[i]) > 0:
                        vector.append(float(a_word_vector[i]))

                word_embedding[word] = vector
                embedding_vector_len = len(vector)

            # the first cell indicates it is unknown word word_embedding['UNK'][0] = 0
            word_embedding['UNK'] = [0 for i in range(embedding_vector_len)]
        self.word_embedding = word_embedding

    def __construct_word_dictionary(self):
        word_counts = [['UNK', -1]]
        words = [normalize_word(word) for word in itertools.chain(*self.sent_words)]
        word_counts.extend(collections.Counter(words).items())

        word_dictionary = dict()
        for word, _ in word_counts:
            if word in self.word_embedding:
                word_dictionary[word] = len(word_dictionary) + 1  # zero will be used for padding

        self.reverse_dictionary = dict(zip(word_dictionary.values(), word_dictionary.keys()))

        self.word_dictionary = word_dictionary


    def __construct_tag_dictionary(self):
        counts = collections.Counter(itertools.chain(*self.sent_tags))
        self.tag_dictionary = dict()
        for tag in counts:
            self.tag_dictionary[tag] = len(self.tag_dictionary) + 1  # +1: 0 is reserved for padding
        self.reverse_dictionary = dict(zip(self.tag_dictionary.values(), self.tag_dictionary.keys()))

    def __set_max_sent_and_word(self):
        sents_len = [len(sent) for sent in self.sent_words]
        words_len = [len(word) for sent in self.sent_words for word in sent]
        self.max_sent_len = max(sents_len)
        self.max_word_len = max(words_len)


    def convert_tags_to_index(self, sent_tags):
        # Convert tags
        Y_train = list()
        for a_tag_seq in sent_tags:
            y_train = []
            for tag in a_tag_seq:
                y_train.append(self.tag_dictionary[tag])
            Y_train.append(y_train)
        return Y_train

    def convert_word_to_index(self, sent_words):
        # Convert samples
        X_word_train = list()
        X_char_train = list()
        for sent in sent_words:
            x_word = []
            x_char = []
            for word in sent:
                chars = [ord(c) for c in normalize_word(word)]
                x_char.append(chars)
                if word in self.word_dictionary and word in self.word_dictionary:
                    index = self.word_dictionary[word]
                else:
                    index = 1  # dictionary['UNK']
                x_word.append(index)
            X_word_train.append(x_word)
            X_char_train.append(x_char)
        return X_word_train, X_char_train

    def get_train_dataset(self):
        try:
            return self.X_word_train, self.X_char_train, self.Y_train
        except:
            self.Y_train = self.convert_tags_to_index(self.sent_tags)
            self.X_word_train, self.X_char_train = self.convert_word_to_index(self.sent_words)

            return self.X_word_train, self.X_char_train, self.Y_train

    def pad_dataset(self, raw_word_train, raw_char_train):
        X_word_padded = pad_sequences(raw_word_train, self.max_sent_len)
        X_char = []
        for i in range(len(raw_char_train)):
            padded_sequence = pad_sequences([[] for k in range(self.max_sent_len - len(raw_char_train[i]))],
                                            maxlen=self.max_word_len).tolist() + \
                              pad_sequences(raw_char_train[i], maxlen=self.max_word_len).tolist()
            X_char.append(padded_sequence)

        X_char_padded = np.array(X_char)
        return X_word_padded, X_char_padded

    def pad_tags(self, raw_tag_train):
        Y_padded_train = pad_sequences(raw_tag_train, self.max_sent_len)
        Y_one_hot = np.zeros((len(Y_padded_train), self.max_sent_len, self.nb_classes))
        for i in range(len(raw_tag_train)):
            Y_one_hot[i] = np_utils.to_categorical(Y_padded_train[i], nb_classes=self.nb_classes)
        return Y_one_hot

    def get_padded_train_datatset(self):
        try:
            return self.X_word_padded, self.X_char_padded, self.Y_one_hot
        except:
            raw_word_train, raw_char_train, raw_tag_train = self.get_train_dataset()
            self.X_word_padded, self.X_char_padded = self.pad_dataset(raw_word_train, raw_char_train)
            self.Y_one_hot = self.pad_tags(raw_tag_train)

            return self.X_word_padded, self.X_char_padded, self.Y_one_hot

    def get_raw(selfs):
        return selfs.sent_words, selfs.sent_tags

    def get_dictionaries_and_max(self):
        return self.word_dictionary, self.tag_dictionary, self.max_sent_len, self.max_word_len

    def get_embedding_weights(self):
        vocab_dim = 301  # dimensionality of your word vectors
        n_symbols = len(self.word_dictionary) + 1  # adding 1 to account for 0th index (for masking)
        embedding_weights = np.zeros((n_symbols, vocab_dim))
        for word, index in self.word_dictionary.items():
            embedding_weights[index, :] = self.word_embedding[word]
        return embedding_weights

def main():
    preprocessor = NnPreprocessor('../data/train.txt', '../data/wordvecs.txt')
    X_word_train, X_char_train, Y_train = preprocessor.get_train_dataset()
    X_word_padded, X_char_padded, Y_padded = preprocessor.get_padded_train_datatset()
    sent_words, sent_tags = preprocessor.get_raw();

    print('sample :\nword = {} \ntags = {} \nchar = {} \n\tAfter padding:\nword = {} \ntags = {} \nchar = {}'.format(
        len(X_word_train), len(Y_train), len(X_char_train), X_word_padded.shape, Y_padded.shape, X_char_padded.shape))

    for i in range(3):
        print(sent_words[i])
        print(sent_tags[i])
        print('sample {}:\nword = {} \ntags = {} \nchar = {} \n\tAfter padding:\nword = {} \ntags = {} \nchar = {}'.format(
              i, X_word_train[i], Y_train[i], X_char_train[i], X_word_padded[i], Y_padded[i], X_char_padded[i]))

def load(train_file, wordvects_file, delimiter='\t'):
    preprocessor = NnPreprocessor(train_file, wordvects_file)

    char_vocab_size = 128 + 1
    word_dictionary, tag_dictionary, max_sent_len, max_word_len = preprocessor.get_dictionaries_and_max()
    word_vocab_size = len(word_dictionary) + 1
    nb_classes = len(tag_dictionary) + 1
    word_embedding_size = 301

    X_word, X_char, Y = preprocessor.get_padded_train_datatset()

    return X_char, X_word, Y, char_vocab_size, max_sent_len, max_word_len, word_embedding_size, nb_classes

if __name__ == "__main__":
    # execute only if run as a script
    main()