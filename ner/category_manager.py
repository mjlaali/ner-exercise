import itertools


class CategoryManager:
    def __init__(self, sent_tags):
        self.__load(sent_tags)

    def __load(self, sent_tags):
        sorted_tags = list(set(itertools.chain(*sent_tags)))
        sorted_tags.sort()

        self.tag_dictionary = dict()
        for tag in sorted_tags:
            self.tag_dictionary[tag] = len(self.tag_dictionary) + 1  # +1: 0 is reserved for padding
        self.reverse_dictionary = dict(zip(self.tag_dictionary.values(), self.tag_dictionary.keys()))

    def convert_to_int(self, sent_tags):
        Y = list()
        for a_tag_seq in sent_tags:
            y = []
            for tag in a_tag_seq:
                y.append(self.tag_dictionary[tag])
            Y.append(y)
        return Y

    def get_num_classes(self):
        return len(self.tag_dictionary) + 1  # +1: 0 is reserved for padding

