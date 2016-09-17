#!/usr/bin/env python
import pickle

import pycrfsuite
import numpy as np
from keras.models import load_model

from ner.crf_train import sent2features

tagger = pycrfsuite.Tagger()
tagger.open('best_model/crfsuite.model')

print('loading the model ...')
with open('best_model/preprocessor.pk', 'rb') as input_file:
    preprocessor = pickle.load(input_file)

model_file = 'best_model/best_model.39.hdf5'
model = load_model(model_file)
_, tag_dict, _, _ = preprocessor.get_dictionaries_and_max()
reversed_tag_dict = dict(zip(tag_dict.values(), tag_dict.keys()))

while True:
    sample_sentence = input('Type a query (type "exit" to exit): \n')
    if sample_sentence == "exit":
        break
    test_sents = [sample_sentence.strip().lower().split()]
    test_words_vecs, test_chars_vecs = preprocessor.get_char_vectors(test_sents)
    X_word_test, X_char_test = preprocessor.pad_dataset(test_words_vecs, test_chars_vecs)
    X_char_test = X_char_test.reshape(X_char_test.shape[0], X_char_test.shape[1] * X_char_test.shape[2])

    Y_predict = model.predict([X_char_test, X_word_test])
    Y_predict = np.argmax(Y_predict, axis=2)

    y_pred = list()
    for i in range(Y_predict.shape[0]):
        y_list = Y_predict[i].tolist()
        y_pred.append([reversed_tag_dict[y] for y in y_list if y > 0])

    print(y_pred[0])