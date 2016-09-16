from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Bidirectional, TimeDistributed
from keras.models import Sequential, load_model, model_from_json
from keras.layers.core import Dense, Dropout, Merge, Reshape, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers.convolutional import Convolution2D, MaxPooling2D
import pickle
from keras.optimizers import RMSprop, Adam
from keras.regularizers import l2

from ner.preprocess_nn import NnPreprocessor


class CNN_LSTM_Model:

    def __init__(self, max_sent_len, max_word_len, char_vocab_size, word_vocab_size, output_vocab_size):
        self.max_sent_len = max_sent_len
        self.max_word_len = max_word_len
        self.char_vocab_size = char_vocab_size
        self.word_vocab_size = word_vocab_size
        self.output_vocab_size = output_vocab_size

    def construct_model2(self, char_embedding_size, nb_filters, word_embedding_size, embedding_weights, lstm_dim):
        model_word = Sequential()
        model_word.add(Embedding(self.word_vocab_size, self.output_vocab_size, input_length=self.max_sent_len))
        model_word.add(TimeDistributed(Dense(100)))
        model_word.add(TimeDistributed(Dense(self.output_vocab_size)))
        model_word.add(Activation('softmax'))
        self.model = model_word
        return model_word

    def construct_model(self, char_embedding_size, nb_filters, word_embedding_size, embedding_weights, lstm_dim):
        filter_length = 3

        model_cnn = Sequential()
        model_cnn.add(Embedding(self.char_vocab_size, char_embedding_size, input_length=self.max_sent_len * self.max_word_len))
        model_cnn.add(Reshape((self.max_sent_len, self.max_word_len, char_embedding_size)))
        model_cnn.add(Convolution2D(nb_filters, 1, filter_length, border_mode='same', dim_ordering='tf'))
        model_cnn.add(MaxPooling2D(pool_size=(1, self.max_word_len), dim_ordering='tf'))
        model_cnn.add(Reshape((self.max_sent_len, nb_filters)))

        model_word = Sequential()
        model_word.add(Embedding(self.word_vocab_size, word_embedding_size, input_length=self.max_sent_len, weights=[embedding_weights]))

        merged = Merge([model_cnn, model_word], mode='concat')

        final_model = Sequential()
        final_model.add(merged)
        final_model.add(Bidirectional(LSTM(output_dim=lstm_dim, activation='sigmoid', inner_activation='hard_sigmoid',
                                 return_sequences=True), merge_mode='concat'))
        final_model.add(Dropout(0.5))
        final_model.add(TimeDistributed(Dense(self.output_vocab_size, W_regularizer=l2(0.01))))
        final_model.add(Activation('softmax'))

        self.model = final_model
        return final_model

    def fit(self, X_word_train, X_char_train, Y_train, batch_size, max_epochs):
        early_stop = EarlyStopping(monitor='val_loss', patience=5, verbose=1)
        save_model = ModelCheckpoint('data/models/best_model.{epoch:02d}.hdf5', save_best_only=True)
        optimizer = Adam()
        self.model.compile(optimizer = optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        X_char_train = X_char_train.reshape(X_char_train.shape[0], X_char_train.shape[1] * X_char_train.shape[2])
        validation_start_idx = int(0.9 * Y_train.shape[0])

        self.model.fit([X_char_train[:validation_start_idx], X_word_train[:validation_start_idx]], Y_train[:validation_start_idx],
                       batch_size=batch_size, nb_epoch=max_epochs,
                       validation_data=([X_char_train[validation_start_idx:], X_word_train[validation_start_idx:]], Y_train[validation_start_idx:]),
                       callbacks=[save_model, early_stop])


import os

def test_model(model_file):
    with open('preprocessor.pk', 'rb') as input:
        preprocessor = pickle.load(input)

    test_file = os.path.join(os.path.dirname(__file__), '../data/test.txt')
    test_sents, test_tags = preprocessor.read_tagged_file(test_file)
    test_words_vecs, test_chars_vecs = preprocessor.convert_word_to_index(test_sents)
    test_tags_vec = preprocessor.convert_tags_to_index(test_tags)
    X_word_test, X_char_test = preprocessor.pad_dataset(test_words_vecs, test_chars_vecs)
    X_char_test = X_char_test.reshape(X_char_test.shape[0], X_char_test.shape[1] * X_char_test.shape[2])
    Y_test = preprocessor.pad_tags(test_tags_vec)

    model = load_model(model_file)
    scores = model.evaluate([X_char_test, X_word_test], Y_test)

    for i in range(len(model.metrics_names)):
        print('{}: {:.4f}'.format(model.metrics_names[i], scores[i]))

def test():
    model_file = os.path.join(os.path.dirname(__file__), '../data/models/best_model.20.hdf5')
    test_model(model_file)

def main():
    train_file = os.path.join(os.path.dirname(__file__), '../data/train.txt')
    wordvects_file = os.path.join(os.path.dirname(__file__), '../data/wordvecs.txt')
    preprocessor = NnPreprocessor(train_file, wordvects_file)

    char_vocab_size = 128 + 1
    word_dictionary, tag_dictionary, max_sent_len, max_word_len = preprocessor.get_dictionaries_and_max()
    word_vocab_size = len(word_dictionary) + 1
    output_vocab_size = len(tag_dictionary) + 1
    print(output_vocab_size)

    model_generator = CNN_LSTM_Model(max_sent_len, max_word_len, char_vocab_size,
                                     word_vocab_size, output_vocab_size)
    X_word_train, X_char_train, Y_train = preprocessor.get_padded_train_datatset()
    model = model_generator.construct_model(100, 100, 301, preprocessor.get_embedding_weights(), 256)
    print(model.summary())
    model_generator.fit(X_word_train, X_char_train, Y_train, 8, 99)

    with open('preprocessor.pk', 'wb') as output:
        pickle.dump(preprocessor, output, pickle.HIGHEST_PROTOCOL)

import time
if __name__ == "__main__":
    main()
    test()
    time.sleep(1) # delays for 5 seconds
