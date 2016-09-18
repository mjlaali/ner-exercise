import itertools

from keras.utils import np_utils


class CategoryManager:
    def __init__(self, sent_tags):
        self.__load(sent_tags)

    def __load(self, sent_tags):
        sorted_tags = list(set(itertools.chain(*sent_tags)))
        sorted_tags.sort()

        self.tag_dictionary = dict()

        for tag in sorted_tags:
            self.tag_dictionary[tag] = len(self.tag_dictionary) + 2
        self.reverse_dictionary = dict(zip(self.tag_dictionary.values(), self.tag_dictionary.keys()))
        self.nb_classes = len(self.tag_dictionary) + 2

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
            Y.append(np_utils.to_categorical(y, nb_classes=self.nb_classes))

        return Y

    def get_num_classes(self):
        return self.nb_classes

