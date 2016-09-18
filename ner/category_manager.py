import itertools

from keras.preprocessing.sequence import pad_sequences
from keras.utils import np_utils
import numpy as np


class CategoryManager:
    def __init__(self, sent_tags):
        self.__load(sent_tags)

    def __load(self, sent_tags):
        sorted_tags = list(set(itertools.chain(*sent_tags)))
        sorted_tags.sort()

        self.tag_dictionary = dict()

        for tag in sorted_tags:
            self.tag_dictionary[tag] = len(self.tag_dictionary) + 1
        self.reverse_dictionary = dict(zip(self.tag_dictionary.values(), self.tag_dictionary.keys()))
        self.nb_classes = len(self.tag_dictionary) + 1

    def pad_tags(self, raw_tag_train, max_sent_len):
        raw_tag_train = self.convert_to_int(raw_tag_train)
        Y_padded_train = pad_sequences(raw_tag_train, max_sent_len)
        Y_one_hot = np.zeros((len(Y_padded_train), max_sent_len, self.nb_classes))
        for i in range(len(raw_tag_train)):
            Y_one_hot[i] = np_utils.to_categorical(Y_padded_train[i], nb_classes=self.nb_classes)
        return Y_one_hot

    # Y_padded_train = pad_sequences(raw_tag_train, self.max_sent_len)
    # Y_one_hot = np.zeros((len(Y_padded_train), self.max_sent_len, self.nb_classes))
    # for i in range(len(raw_tag_train)):
    #     Y_one_hot[i] = np_utils.to_categorical(Y_padded_train[i], nb_classes=self.nb_classes)
    # return Y_one_hot

    def convert_to_int(self, sent_tags):
        Y = list()
        for a_tag_seq in sent_tags:
            y = []
            for tag in a_tag_seq:
                y.append(self.tag_dictionary[tag])
            Y.append(y)

        return Y

    def pad_y(self, sent_tags, max_sent_len):
        sent_tags_int = self.convert_to_int(sent_tags)
        Y_int = pad_sequences(sent_tags_int, max_sent_len)
        Y = list()
        for y_seq in Y_int:
            y = np_utils.to_categorical(y_seq, nb_classes=self.nb_classes)
            Y.append(y.tolist())

        return np.array(Y)

    def get_num_classes(self):
        return self.nb_classes

