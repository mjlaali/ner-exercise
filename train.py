#!/usr/bin/env python
import os

from ner.cnn_lstm_train import train_model

if __name__ == "__main__":
    train_file = os.path.join(os.path.dirname(__file__), 'data/coling2016/train')
    train_feature_file = os.path.join(os.path.dirname(__file__), 'data/coling2016/result/train.feats')
    wordvects_file = os.path.join(os.path.dirname(__file__), 'data/coling2016/glove.twitter.27B.200d.txt')
    test_file = os.path.join(os.path.dirname(__file__), 'data/coling2016/dev')
    delimiter = ' '

    train_model(train_file, wordvects_file, train_feature_file, delimiter=delimiter)

