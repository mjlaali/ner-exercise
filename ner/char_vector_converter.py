from keras.preprocessing.sequence import pad_sequences
import numpy as np


class CharVectorConverter:
    def __init__(self):
        pass


    def get_char_vectors(self, sent_words, max_word_len=-1):
        # Convert samples
        dataset_max_word_len = 0
        dataset_max_sent_len = 0
        dataset_max_char_value = 0
        sents_words_chars = list()
        for sent in sent_words:
            sent_words_chars = []
            if len(sent) > dataset_max_sent_len:
                dataset_max_sent_len = len(sent)
            for word in sent:
                if max_word_len != -1:
                    word = CharVectorConverter.shrink(word, max_word_len)
                word_chars = [ord(c) for c in word]
                if max(word_chars) > dataset_max_char_value:
                    dataset_max_char_value = max(word_chars)

                if len(word_chars) > dataset_max_word_len:
                    dataset_max_word_len = len(word_chars)

                sent_words_chars.append(word_chars)
            sents_words_chars.append(sent_words_chars)

        self.char_vocab_size = dataset_max_char_value + 1
        return sents_words_chars, dataset_max_sent_len, dataset_max_word_len

    def get_char_vocab_size(self):
        return self.char_vocab_size

    @staticmethod
    def shrink(str, max_len):
        if (len(str) > max_len):
            prefix = max_len / 2
            postfix = len(str) - max_len + prefix
            str = str[:prefix] + str[postfix:]
        return str