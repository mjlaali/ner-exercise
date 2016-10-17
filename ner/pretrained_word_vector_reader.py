import csv


class PretrainedWordVectorReader:
    UNK = 'UNK'

    def __init__(self, word_vector_file):
        self.word_vector_file = word_vector_file

    def load(self, words, delimiter='\t'):
        print('start loading word vectors ...')

        words = set([word.lower() for word in words])
        with open(self.word_vector_file, 'r') as csvfile:
            word_vectors = csv.reader(csvfile, delimiter=delimiter, quoting=csv.QUOTE_NONE)
            word_embedding = {}
            len_embedding_vector = 0
            idx = 0
            for a_word_vector in word_vectors:
                idx += 1
                word = a_word_vector[0]
                if word in words:
                    vector = [0]  # first cell is a flag for seen words
                    for i in range(1, len(a_word_vector)):
                        if len(a_word_vector[i]) > 0:
                            vector.append(float(a_word_vector[i]))

                    word_embedding[word] = vector
                    if len_embedding_vector == 0:
                        len_embedding_vector = len(vector)
                    elif len_embedding_vector != len(vector):
                        print('Error, the word has different length: {} -> {} != {}'.format(word, len_embedding_vector, len(vector)))
                if idx % 10000 == 0:
                    print('{} words has been read'.format(idx))

            print('The length of word embedding is {}'.format(len_embedding_vector))
            # the first cell indicates it is unknown word word_embedding['UNK'][0] = 1
            word_embedding[self.UNK] = [0 for i in range(len_embedding_vector)]
            word_embedding[self.UNK][0] = 1
            self.len_embedding_vector = len_embedding_vector
        self.word_embedding = word_embedding
        return self.word_embedding

    def get_len_embedding_vector(self):
        return self.len_embedding_vector

    def convert_dataset(self, sents):
        sent_vectors = list()
        for sent in sents:
            sent_vector = list()
            for word in sent:
                word = word.lower()
                if word in self.word_embedding:
                    sent_vector.append(self.word_embedding[word])
                else:
                    sent_vector.append(self.word_embedding[self.UNK])
            sent_vectors.append(sent_vector)

        return sent_vectors