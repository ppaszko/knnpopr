import numpy as np
from DataImport import *


class Perceptron:
    def __init__(self):
        self.threshold = 0
        self.weights = np.random.rand(train_data.features.shape[1], 1)

    def prediction(self, features_row):
        label_prediction = np.matmul(self.weights.T, features_row) + self.threshold
        label_prediction = int(label_prediction > 0)

        return label_prediction

    def train(self, train_data, max_iter=200, learning_rate=0.1, verbose=True):

        for i in range(max_iter):
            errors_count = 0
            for features_row, label_value in zip(train_data.features, train_data.labels):
                true_label = int(label_value[0])
                label_prediction = self.prediction(features_row)

                if true_label - label_prediction != 0:
                    features_row = features_row.reshape(len(features_row), 1)
                    # need to reshape from (8, ) to (8,1). why extracting rows from array gives something that is not a vector?

                    self.weights = np.add(self.weights,
                                          (features_row * learning_rate * (true_label - label_prediction)))
                    self.threshold+=learning_rate*(true_label - label_prediction)
                    errors_count += 1


            if verbose:
                error_rate = abs(errors_count) / len(train_data.features)
                print('Iteration :{}, accuracy: {}, error rate: {} \n'.format(i, 1 - error_rate, error_rate,))

    def test(self, test_data):
        errors_count = 0
        for features_row, label_value in zip(test_data.features, test_data.labels):
            true_label = int(label_value[0])
            label_prediction = self.prediction(features_row)
            if true_label - label_prediction != 0:
                errors_count += 1

        error_rate = abs(errors_count) / len(test_data.features)
        return {'error rate': error_rate,
                'accuracy': 1 - error_rate}


if __name__ == '__main__':
    data = DataHandlerCSV()
    data.import_data_fromcsv('sample1.csv')
    data.get_labels_and_normalize_centralize_features()
    train_data, test_data = data.train_and_test_set_split(0.8)
    mad_mind = Perceptron()
    mad_mind.train(train_data, learning_rate=0.1)

    print('Test results\n', mad_mind.test(test_data))
    print('\n\n')

    data = DataHandlerCSV()
    data.import_data_fromcsv('sample2.csv')
    data.get_labels_and_normalize_centralize_features()
    train_data, test_data = data.train_and_test_set_split(0.8)
    mad_mind = Perceptron()
    mad_mind.train(train_data, learning_rate=0.01, verbose=False)

    print('Test results\n', mad_mind.test(test_data))
    print('\n\n')

    data = DataHandlerCSV()
    data.import_data_fromcsv('sample3.csv')
    data.get_labels_and_normalize_centralize_features()
    train_data, test_data = data.train_and_test_set_split(0.8)
    mad_mind = Perceptron()
    mad_mind.train(train_data, learning_rate=0.01, verbose=False)

    print('Test results\n', mad_mind.test(test_data))
    print('\n\n')

