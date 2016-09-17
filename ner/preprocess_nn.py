import collections
import csv
import itertools

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
                                        maxlen=max_dim_2, dtype=dtype).tolist() + \
                          pad_sequences(sequence[i], maxlen=max_dim_2, dtype=dtype).tolist()
        X.append(padded_sequence)
    return np.array(X)

class NnPreprocessor:

    def __init__(self, train_file):
        self.sent_words, self.sent_tags = NnPreprocessor.read_tagged_file(train_file)
        pass

    @staticmethod
    def read_tagged_file(file_path):
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
                    if len(words) < 40:
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


    def convert_tags_to_index(self, sent_tags):
        # Convert tags
        Y_train = list()
        for a_tag_seq in sent_tags:
            y_train = []
            for tag in a_tag_seq:
                y_train.append(self.tag_dictionary[tag])
            Y_train.append(y_train)
        return Y_train


    def get_train_dataset(self):
        return self.sent_words, self.sent_tags


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
        # for word, index in self.word_dictionary.items():
        #     embedding_weights[index, :] = self.word_embedding[word]
        return embedding_weights

def main():
    train_dataset_file = '../data/coling2016/train'
    word_vector_file = '../data/coling2016/glove.twitter.27B.200d.txt'

    preprocessor = NnPreprocessor(train_dataset_file)
    sent_words, sent_tags = preprocessor.get_train_dataset()
    sent_words_chars, max_sent_len, max_word_len = CharVectorConverter().get_char_vectors(sent_words)
    X_chars = pad_sequences_2D(sent_words_chars, max_sent_len, max_word_len)

    reader = PretrainedWordVectorReader(word_vector_file)
    reader.load(set(itertools.chain(*sent_words)), delimiter=' ')
    sent_words_int = reader.convert_dataset(sent_words)
    X_words = pad_sequences_2D(sent_words_int, max_sent_len, reader.get_len_embedding_vector(), dtype='float32')

    category_manager = CategoryManager(sent_tags)
    sent_tags_int = category_manager.convert_to_int(sent_tags)
    Y = pad_sequences(sent_tags_int, max_sent_len)

    print('sample :\nword = {} \ntags = {} \nchar = {} \n\tAfter padding:\nword = {} \ntags = {} \nchar = {}'.format(
        len(X_words), len(Y), len(X_chars), X_words.shape, Y.shape, X_chars.shape))

    for i in range(1):
        print(sent_words[i])
        print(sent_tags[i])
        print('sample {}:\nword = {} \ntags = {} \nchar = {} \n\tAfter padding:\nword = {} \ntags = {} \nchar = {}'.format(
              i, sent_words_int[i], sent_tags_int[i], sent_words_chars[i], X_words[i], Y[i], X_chars[i]))

if __name__ == "__main__":
    # execute only if run as a script
    main()