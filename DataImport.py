import csv
import numpy as np


class DataHandlerCSV:

    def __init__(self, data=None, labels_column_number=None):
        self.data = data
        if labels_column_number is None:
            self.labels = None
            self.features = None
        else:
            self.labels = self.get_labels(labels_column_number)
            self.features = self.get_features(labels_column_number)

    def get_labels_and_normalize_centralize_features(self, labels_column_number=-1, new_min=-1, new_max=1):
        self.get_labels(labels_column_number)
        self.get_features(labels_column_number)
        self.all_rows_centralize()
        self.all_rows_normalize(new_min, new_max)

    def import_data_fromcsv(self, path, delimiter=';'):
        csvfile = open(path)
        with csvfile:
            data = csv.reader(csvfile, delimiter=delimiter)
            data = [d for d in data]
            self.data = data
            self.data = np.array(self.data)
        return self.data

    def get_labels(self, column_number=-1):
        self.labels = np.array(self.data[:, column_number])
        #self.labels = self.labels.reshape((len(self.labels), 1))
        print(np.shape(self.labels))
        # need to reshape. why numpy array with one didnt create a vector?

        return self.labels

    def get_features(self, labels_column_number=-1):
        data = np.array(self.data, dtype=float)
        self.features = np.delete(data, labels_column_number, 1)
        return self.features

    def column_centre(self, column_number=1):
        self.features[:, column_number] = self.features[:, column_number]
        self.features[:, column_number] = self.features[:, column_number] - np.mean(self.features[:, column_number])
        return self.features

    def column_normalize(self, column_number=-1, new_min=-1, new_max=1):
        # min max. normalization default [-1,1]
        self.features[:, column_number] = (
                (self.features[:, column_number] - min(self.features[:, column_number])) / (
                    max(self.features[:, column_number]) - min(self.features[:, column_number])) * (
                            new_max - new_min) + new_min)
        return self.features

    def all_rows_normalize(self, new_min=-1, new_max=1):
        for i in range(self.features.shape[1]):
            self.features = self.column_normalize(i, new_min, new_max)
        return self.features

    def all_rows_centralize(self):
        for i in range(self.features.shape[1]):
            self.features = self.column_centre(i)

        return self.features

    def train_and_test_set_split(self, training_set_size=0.8):
        features_and_labels = np.append(self.features, self.labels, axis=1)
        np.random.shuffle(features_and_labels)
        n_training_rows = int(self.features.shape[0] * training_set_size)
        train_set = features_and_labels[:n_training_rows]
        test_set = features_and_labels[n_training_rows:]
        return DataHandlerCSV(train_set, -1), DataHandlerCSV(test_set, -1)

