#!/usr/bin/env python
import pycrfsuite

from ner.crf_train import sent2features

tagger = pycrfsuite.Tagger()
tagger.open('best_model/crfsuite.model')

while True:
    sample_sentence = input('Type a query (type "exit" to exit): \n')
    if sample_sentence == "exit":
        break
    sample_input = sample_sentence.strip().lower().split()
    X_test = sent2features(sample_input)
    print(tagger.tag(X_test))