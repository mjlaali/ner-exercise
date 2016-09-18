import csv
import os

from keras.utils import np_utils
import numpy as np


class FeatureReader:

    def __init__(self, feature_file):
        self.feature_file = feature_file
        self.dataset, self.features_values = FeatureReader.load(feature_file)
        self.__init_feature_idx()

    @staticmethod
    def load(feature_file):
        print('start loading word features ...')

        idx = 0
        with open(feature_file, 'r') as csvfile:
            all_features = csv.reader(csvfile, delimiter='\t', quoting=csv.QUOTE_NONE)
            features_values = dict()

            dataset = list()
            sent_features = list()
            for features_line in all_features:
                word_features = dict()
                end_seq = False
                for a_word_feature in features_line[1:]:
                    vals = a_word_feature.split('=', 1)
                    word_feature_name = vals[0]
                    if word_feature_name in features_values:
                        feature_values = features_values[word_feature_name]
                    else:
                        feature_values = set()
                        features_values[word_feature_name] = feature_values

                    if len(vals) == 2:
                        word_feature_val = vals[1]
                    else:
                        word_feature_val = 1

                    feature_values.add(word_feature_val)

                    word_features[word_feature_name] = word_feature_val

                    if word_feature_name == "__EOS__":
                        end_seq = True

                sent_features.append(word_features)
                if end_seq:
                    dataset.append(sent_features)
                    sent_features = list()
                idx += 1
                if idx % 1000 == 0:
                    print('{} features has been read'.format(idx))

        return dataset, features_values

    def get_dataset(self):
        return self.dataset

    def __init_feature_idx(self):
        feature_value_idx = dict()
        self.sorted_feature_names = list(self.features_values.keys())
        self.sorted_feature_names.sort()

        for feature_name in self.sorted_feature_names:
            feature_values = self.features_values[feature_name]

            value_idx = dict()
            feature_values = list(feature_values)
            feature_values.sort()
            for feature_value in feature_values:
                value_idx[feature_value] = len(value_idx) + 1

            feature_value_idx[feature_name] = value_idx
        self.feature_value_idx = feature_value_idx

    def get_feature_vector(self, max_sent_len, dataset=None):
        self.ignored_features = set()
        print('start constructing feature vectors ...')
        if dataset == None:
            dataset = self.dataset

        dataset_vector = list()

        features_len = -1;
        idx = 0
        for sent in dataset:
            sent_vector = list()
            if len(sent) > max_sent_len:
                sent = sent[:max_sent_len]

            for word in sent:
                feature_vecs = self.convert_word_features(word)
                if features_len == -1:
                    features_len = len(feature_vecs)
                    zeros = [0 for i in range(features_len)]
                elif features_len != len(feature_vecs):
                    print('Error: Features do not have the same length')

                sent_vector.append(feature_vecs)

            sent_vector = [zeros for i in range(max_sent_len - len(sent_vector))] + sent_vector
            dataset_vector.append(sent_vector)
            idx += 1
            if idx % 10 == 0:
                # print(idx)
                pass

        print(self.ignored_features)
        print('Done!')

        return dataset_vector


    def convert_word_features(self, word_features):
        word_features_idx = list()
        for feature_name in self.sorted_feature_names:
            feature_values_len = len(self.features_values[feature_name])
            if feature_values_len > 100:
                self.ignored_features.add(feature_name)
                continue
            if feature_name in word_features:
                if feature_values_len == 1:
                    word_features_idx.append(1)
                else:
                    feature_value = word_features[feature_name]
                    feature_value_idx = self.feature_value_idx[feature_name][feature_value]
                    feature_value_vec = [0 for i in range(feature_values_len + 1)]
                    feature_value_vec[feature_value_idx] = 1
                    word_features_idx.extend(feature_value_vec)
            else:
                if feature_values_len == 1:
                    word_features_idx.append(1)
                else:
                    word_features_idx.extend([0 for i in range(feature_values_len + 1)])

        return word_features_idx

def main():
    feature_file = os.path.join(os.path.dirname(__file__), '../data/coling2016/result/dev.feats')
    feature_reader = FeatureReader(feature_file=feature_file)
    feature_vector = feature_reader.get_feature_vector(40)
    print(np.array(feature_vector).shape)

if __name__ == "__main__":
    # execute only if run as a script
    main()