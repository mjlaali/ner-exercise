from keras.preprocessing.sequence import pad_sequences
import numpy as np


class CharVectorConverter:
    def __init__(self):
        pass

    def get_char_vectors(self, sent_words):
        # Convert samples
        max_word_len = 0
        max_sent_len = 0
        sents_words_chars = list()
        for sent in sent_words:
            sent_words_chars = []
            if len(sent) > max_sent_len:
                max_sent_len = len(sent)
            for word in sent:
                word_chars = [ord(c) for c in word]
                if len(word_chars) > max_word_len:
                    max_word_len = len(word_chars)

                sent_words_chars.append(word_chars)
            sents_words_chars.append(sent_words_chars)
        return sents_words_chars, max_sent_len, max_word_len

