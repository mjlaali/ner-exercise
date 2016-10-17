import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Bidirectional, TimeDistributed, Convolution1D, MaxPooling1D
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.core import Dense, Dropout, Merge, Reshape, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.models import Sequential, load_model
from keras.optimizers import Adam, RMSprop
from keras.regularizers import l2

import ner
import pickle
from ner.category_manager import CategoryManager
from ner.preprocess_nn import NnPreprocessor
from ner.report import bio_classification_report
import os.path

class CNN_LSTM_Model:

    def __init__(self, max_sent_len, max_word_len, char_vocab_size, nb_classes, word_embedding_size, nb_features):
        self.max_sent_len = max_sent_len
        self.max_word_len = max_word_len
        self.char_vocab_size = char_vocab_size
        self.nb_classes = nb_classes
        self.word_embedding_size = word_embedding_size
        self.nb_features = nb_features


    def construct_model(self, char_embedding_size, nb_filters, lstm_dim, filter_length = 3):

        model_cnn = Sequential()
        model_cnn.add(Embedding(self.char_vocab_size, char_embedding_size, input_length=self.max_sent_len * self.max_word_len))
        model_cnn.add(Reshape((self.max_sent_len, self.max_word_len, char_embedding_size)))
        model_cnn.add(Convolution2D(nb_filters, 1, filter_length, border_mode='same', dim_ordering='tf'))
        model_cnn.add(MaxPooling2D(pool_size=(1, self.max_word_len), dim_ordering='tf'))
        model_cnn.add(Reshape((self.max_sent_len, nb_filters)))

        model_word = Sequential()
        model_word.add(TimeDistributed(Dense(nb_filters, W_regularizer=l2(0.01), activation='sigmoid'),
                                             input_shape=(self.max_sent_len, self.word_embedding_size)))

        model_feature = Sequential()
        model_feature.add(TimeDistributed(Dense(nb_filters, W_regularizer=l2(0.01), activation='sigmoid'),
                                             input_shape=(self.max_sent_len, self.nb_features)))

        merged = Merge([model_cnn, model_word, model_feature], mode='sum')

        final_model = Sequential()
        final_model.add(merged)
        # final_model = model_cnn
        final_model.add(Bidirectional(LSTM(output_dim=lstm_dim, activation='sigmoid', inner_activation='hard_sigmoid',
                                 return_sequences=True), merge_mode='sum'))
        final_model.add(Dropout(0.5))
        final_model.add(TimeDistributed(Dense(self.nb_classes, W_regularizer=l2(0.01))))
        final_model.add(Activation('softmax'))

        self.model = final_model
        return final_model

    def fit(self, X_char, X_word, X_features, Y, batch_size, max_epochs):
        early_stop = EarlyStopping(monitor='val_loss', patience=20, verbose=1)
        save_model = ModelCheckpoint('data/coling-tag/best_model.{epoch:02d}.hdf5', save_best_only=True)
        optimizer = Adam()
        self.model.compile(optimizer = optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        X_char = X_char.reshape(X_char.shape[0], X_char.shape[1] * X_char.shape[2])
        validation_start_idx = int(0.9 * Y.shape[0])
        if validation_start_idx == 0:
            X_train = [X_char, X_word, X_features]
            # X_train = [X_char]
            Y_train = Y
            validation_data = None
            callbacks = []
        else:
            X_train = [X_char[:validation_start_idx], X_word[:validation_start_idx], X_features[:validation_start_idx]]
            Y_train = Y[:validation_start_idx]
            validation_data = ([X_char[validation_start_idx:], X_word[validation_start_idx:], X_features[validation_start_idx:]],
                               Y[validation_start_idx:])
            callbacks = [save_model, early_stop]
        self.model.fit(X_train, Y_train,
                       batch_size=batch_size, nb_epoch=max_epochs,
                       validation_data=validation_data,
                       callbacks=callbacks)


import os

def test_model(model_file, test_file, wordvects_file, train_file, delimiter='\t'):
    # train_file = os.path.join(os.path.dirname(__file__), '../data/coling2016/train')
    # trian_sents, y_train = preprocessor.read_tagged_file(train_file)
    # train_words_vecs, train_chars_vecs = preprocessor.convert_word_to_index(trian_sents)
    # expected_word_idx, _, _ = preprocessor.get_train_dataset()
    #
    # print(train_words_vecs[10])
    # print(expected_word_idx[10])
    train_dataset = ner.preprocess_nn.load(train_file, wordvects_file, load_word_vectors=False)
    max_sent_len = train_dataset['max_sent_len']
    max_word_len = train_dataset['max_word_len']
    test_dataset = ner.preprocess_nn.load(test_file, wordvects_file,
                                          max_sent_len=max_sent_len, max_word_len=max_word_len, delimiter=delimiter)
    X_chars = test_dataset['X_chars']
    X_words = test_dataset['X_words']
    print(X_chars.shape)
    X_chars = X_chars.reshape(X_chars.shape[0], X_chars.shape[1] * X_chars.shape[2])

    train_sent_tags = train_dataset['sent_tags']
    category_manager = CategoryManager(train_sent_tags)

    model = load_model(model_file)
    print(model.summary())

    Y_predict = model.predict([X_chars, X_words])
    y_pred = category_manager.convert_to_tag(Y_predict)

    test_sent_tags = test_dataset['sent_tags']
    for i in range(10):
        print('{}\n{}\n'.format(test_sent_tags[i], y_pred[i]))

    print(bio_classification_report(test_sent_tags, y_pred))


def train_model(train_file, wordvects_file, train_feature_file, delimiter='\t'):
    training_dataset_file = 'training-dataset.pk'
    if os.path.isfile(training_dataset_file):
        print('start loadding the dataset ...')
        with open(training_dataset_file, 'rb') as input:
            train_dataset = pickle.load(input)
    else:
        print('start building the dataset ...')
        train_dataset = ner.preprocess_nn.load(train_file, wordvects_file,
                        feature_file=train_feature_file, delimiter=delimiter)
        with open(training_dataset_file, 'wb') as output:
            pickle.dump(train_dataset, output, pickle.HIGHEST_PROTOCOL)

    char_vocab_size = train_dataset['char_vocab_size']
    nb_classes = train_dataset['nb_classes']
    max_sent_len = train_dataset['max_sent_len']
    max_word_len = train_dataset['max_word_len']
    word_embedding_size = train_dataset['word_embedding_size']
    nb_features = train_dataset['crf_nb_features']
    print('Chars={}, Classes={}, Max sent len={}, Max word len={}, Features={}'.format(
        char_vocab_size, nb_classes, max_sent_len, max_word_len, nb_features))

    model_generator = CNN_LSTM_Model(max_sent_len=max_sent_len, max_word_len=max_word_len, char_vocab_size=char_vocab_size,
                                     nb_classes=nb_classes, word_embedding_size=word_embedding_size, nb_features=nb_features)

    print('start constructing the model ...')
    model = model_generator.construct_model(char_embedding_size=50, nb_filters=2000, lstm_dim=nb_classes*4, filter_length=4)
    print(model.summary())
    print('start learning the model ...')
    indexes = np.array([1])

    X_chars = train_dataset['X_chars']#[indexes]
    X_words = train_dataset['X_words']#[indexes]
    X_features = train_dataset['crf_features']#[indexes]
    Y = train_dataset['Y']#[indexes]
    model_generator.fit(X_char=X_chars, X_word=X_words, X_features=X_features, Y=Y, batch_size=8, max_epochs=1000)

import time

if __name__ == "__main__":
    train_file = os.path.join(os.path.dirname(__file__), '../data/coling2016/train')
    train_feature_file = os.path.join(os.path.dirname(__file__), '../data/coling2016/result/train.feats')
    wordvects_file = os.path.join(os.path.dirname(__file__), '../data/coling2016/glove.twitter.27B.200d.txt')
    test_file = os.path.join(os.path.dirname(__file__), '../data/coling2016/dev')
    delimiter = ' '


    # train_file = os.path.join(os.path.dirname(__file__), '../data/maluuba/train.txt')
    # wordvects_file = os.path.join(os.path.dirname(__file__), '../data/maluuba/wordvecs.txt')
    # test_file = os.path.join(os.path.dirname(__file__), '../data/maluuba/test.txt')
    # delimiter = '\t'

    idx = 12
    test_model_file = os.path.join(os.path.dirname(__file__), '../data/coling-tag/best_model.{:02d}.hdf5'.format(idx))

    train_model(train_file, wordvects_file, train_feature_file, delimiter=delimiter)
    test_model(test_model_file, test_file, wordvects_file, train_file,  delimiter=delimiter)
    time.sleep(1) # delays to close the tensorflow session
