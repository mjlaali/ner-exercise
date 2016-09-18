import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Bidirectional, TimeDistributed, Convolution1D, MaxPooling1D
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.core import Dense, Dropout, Merge, Reshape, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.models import Sequential, load_model
from keras.optimizers import Adam
from keras.regularizers import l2

import ner
from ner.preprocess_nn import NnPreprocessor
from ner.report import bio_classification_report


class CNN_LSTM_Model:

    def __init__(self, max_sent_len, max_word_len, char_vocab_size, nb_classes, word_embedding_size):
        self.max_sent_len = max_sent_len
        self.max_word_len = max_word_len
        self.char_vocab_size = char_vocab_size
        self.nb_classes = nb_classes
        self.word_embedding_size = word_embedding_size

    def construct_model2(self, char_embedding_size, nb_filters, lstm_dim):
        filter_length = 3
        model_cnn = Sequential()
        model_cnn.add(Embedding(self.char_vocab_size, char_embedding_size, input_length=self.max_sent_len * self.max_word_len, mask_zero=False))
        model_cnn.add(Convolution1D(nb_filters, filter_length * self.max_word_len, border_mode='same'))
        model_cnn.add(MaxPooling1D(pool_length=self.max_word_len))

        final_model = model_cnn
        final_model.add(TimeDistributed(Dense(self.nb_classes)))
        final_model.add(Activation('softmax'))
        self.model = final_model
        return final_model

    def construct_model(self, char_embedding_size, nb_filters, lstm_dim):
        filter_length = 3

        model_cnn = Sequential()
        model_cnn.add(Embedding(self.char_vocab_size, char_embedding_size, input_length=self.max_sent_len * self.max_word_len))
        model_cnn.add(Reshape((self.max_sent_len, self.max_word_len, char_embedding_size)))
        model_cnn.add(Convolution2D(nb_filters, 1, filter_length, border_mode='same', dim_ordering='tf'))
        model_cnn.add(MaxPooling2D(pool_size=(1, self.max_word_len), dim_ordering='tf'))
        model_cnn.add(Reshape((self.max_sent_len, nb_filters)))

        model_word = Sequential()
        model_word.add(TimeDistributed(Dense(nb_filters, W_regularizer=l2(0.01), activation='sigmoid'),
                                             input_shape=(self.max_sent_len, self.word_embedding_size)))

        merged = Merge([model_cnn, model_word], mode='concat')

        final_model = Sequential()
        final_model.add(merged)
        final_model = model_cnn
        final_model.add(Bidirectional(LSTM(output_dim=lstm_dim, activation='sigmoid', inner_activation='hard_sigmoid',
                                 return_sequences=True), merge_mode='concat'))
        final_model.add(Dropout(0.5))
        final_model.add(TimeDistributed(Dense(self.nb_classes, W_regularizer=l2(0.01))))
        final_model.add(Activation('softmax'))

        self.model = final_model
        return final_model

    def fit(self, X_char, X_word, Y, batch_size, max_epochs):
        early_stop = EarlyStopping(monitor='val_loss', patience=5, verbose=1)
        save_model = ModelCheckpoint('data/coling-tag/best_model.{epoch:02d}.hdf5', save_best_only=True)
        optimizer = Adam()
        self.model.compile(optimizer = optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        X_char = X_char.reshape(X_char.shape[0], X_char.shape[1] * X_char.shape[2])
        validation_start_idx = int(0.9 * Y.shape[0])
        if validation_start_idx == 0:
            # X_train = [X_char, X_word]
            X_train = [X_char]
            Y_train = Y
            validation_data = None
            callbacks = []
        else:
            X_train = [X_char[:validation_start_idx], X_word[:validation_start_idx]]
            Y_train = Y[:validation_start_idx]
            validation_data = ([X_char[validation_start_idx:], X_word[validation_start_idx:]], Y[validation_start_idx:])
            callbacks = [save_model, early_stop]
        self.model.fit(X_train, Y_train,
                       batch_size=batch_size, nb_epoch=max_epochs,
                       validation_data=validation_data,
                       callbacks=callbacks)


import os

def test_model(model_file, test_file, wordvects_file):
    pass
    # train_file = os.path.join(os.path.dirname(__file__), '../data/coling2016/train')
    # trian_sents, y_train = preprocessor.read_tagged_file(train_file)
    # train_words_vecs, train_chars_vecs = preprocessor.convert_word_to_index(trian_sents)
    # expected_word_idx, _, _ = preprocessor.get_train_dataset()
    #
    # print(train_words_vecs[10])
    # print(expected_word_idx[10])

    # X_char, X_word, Y, char_vocab_size, max_sent_len, max_word_len, nb_classes = ner.preprocess_nn.load(test_file,
    #                                                                                                     wordvects_file)
    # X_char = X_char.reshape(X_char.shape[0], X_char.shape[1] * X_char.shape[2])
    # reversed_tag_dict = dict(zip(tag_dict.values(), tag_dict.keys()))
    #
    # model = load_model(model_file)
    # Y_predict = model.predict([X_char_test, X_word_test])
    # Y_predict = np.argmax(Y_predict, axis=2)
    #
    # y_pred = list()
    # for i in range(Y_predict.shape[0]):
    #     y_list = Y_predict[i].tolist()
    #     y_pred.append([reversed_tag_dict[y] for y in y_list if y > 0])
    #
    # for i in range(10):
    #     print('{}\n{}\n'.format(y_test[i], y_pred[i]))
    #
    # print(bio_classification_report(y_test, y_pred))


def train_model(train_file, wordvects_file):
    X_char, X_word, Y, char_vocab_size, max_sent_len, max_word_len, word_embedding_size, nb_classes = \
        ner.preprocess_nn.load(train_file, wordvects_file, delimiter='\t')
    print('Chars={}, Classes={}, Max sent len={}, Max word len={}'.format(char_vocab_size, nb_classes,
                                                                       max_sent_len, max_word_len))

    model_generator = CNN_LSTM_Model(max_sent_len=max_sent_len, max_word_len=max_word_len, char_vocab_size=char_vocab_size,
                                     nb_classes=nb_classes, word_embedding_size=word_embedding_size)

    print('start constructing the model ...')
    model = model_generator.construct_model(char_embedding_size=100, nb_filters=100, lstm_dim=256)
    print(model.summary())
    print('start learning the model ...')
    indexes = np.array([1])
    model_generator.fit(X_char=X_char[indexes], X_word=X_word[indexes], Y=Y[indexes], batch_size=8, max_epochs=1000)

    X_char = X_char.reshape(X_char.shape[0], X_char.shape[1] * X_char.shape[2])
    Y_predict = model.predict(X_char[indexes])
    Y_predict = np.argmax(Y_predict, axis=2)

    print(Y_predict)
    print(np.argmax(Y[indexes], axis=2))

import time

if __name__ == "__main__":
    # train_file = os.path.join(os.path.dirname(__file__), '../data/coling2016/train')
    # wordvects_file = os.path.join(os.path.dirname(__file__), '../data/coling2016/glove.twitter.27B.200d.txt')
    # test_file = os.path.join(os.path.dirname(__file__), '../data/coling2016/dev')

    train_file = os.path.join(os.path.dirname(__file__), '../data/maluuba/train.txt')
    wordvects_file = os.path.join(os.path.dirname(__file__), '../data/maluuba/wordvecs.txt')
    test_file = os.path.join(os.path.dirname(__file__), '../data/coling2016/dev')

    idx = 37
    test_model_file = os.path.join(os.path.dirname(__file__), '../data/coling-tag/best_model.{:02d}.hdf5'.format(idx))

    train_model(train_file, wordvects_file)
    #test_model(test_model_file, test_file)
    time.sleep(1) # delays to close the tensorflow session
