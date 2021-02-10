import argparse
import numpy as np


class FileLoader:

    def __init__(self, path_first, path_second=None):
        self.train_set = None
        self.test_set = None
        self.data = None
        self.prediction_set = None
        if path_second:
            self.load_train_test(path_first, path_second)
        elif path_first:
            self.data = self.load_data(path_first)

    @staticmethod
    def load_data(path):
        loaded_data = np.loadtxt(path, delimiter=' ', dtype='float')
        return loaded_data

    def train_and_test_split(self, ratio=0.8):
        if self.data is not None:
            np.random.shuffle(self.data)
            n_training_rows = int(self.data.shape[0] * ratio)
            self.train_set = self.data[:n_training_rows]
            self.test_set = self.data[n_training_rows:]

    def load_train_test(self, train_path, test_path):
        self.train_set = self.load_data(train_path)
        self.test_set = self.load_data(test_path)

    def load_prediction(self, prediction_path=None):
        if prediction_path is not None:
            self.prediction_set = self.load_data(prediction_path)

    def train_features(self):
        return np.delete(self.train_set, -1, 1)

    def test_features(self):
        return np.delete(self.test_set, -1, 1)

    def train_labels(self):
        return np.array(self.train_set[:, -1])

    def test_labels(self):
        return np.array(self.test_set[:, -1])


class KNearestNeighbor:
    def __init__(self):
        self.x_train = None
        self.y_train = None

    def train(self, x, y):
        self.x_train = x
        self.y_train = y

    def predict(self, x_test, k):
        distances = self.compute_distance_one_loop(x_test)
        return self.my_predict_labels(distances, k)



    def compute_distance_one_loop(self, x_test):
        """Compute distance to all points at once. Simply with abs function. np.sum is needed for proper numpy broadcasting (need to specify axis)"""

        distances = np.zeros((x_test.shape[0], self.x_train.shape[0]))

        for row_number in range(x_test.shape[0]):
            distances[row_number, :] = np.sum(abs((self.x_train - x_test[row_number, :])), axis=1)

        return distances

    def my_predict_labels(self, distances, k):

        y_pred = np.zeros(distances.shape[0])
        for row_number in range(distances.shape[0]):
            y_pred[row_number] = self.prediction_single_label(distances, k, row_number)

        return y_pred

    def prediction_single_label(self, distances, k, row_number):

        #argsort returns sorted index numbers
        y_indices = np.argsort(distances[row_number, :])

        # dict with classes and their neighbours indexes (need for pair handling)
        indexes_dist_classes = {}
        counter = 0

        for k_ind in range(k):
            #
            class_recognized = self.y_train[y_indices[k_ind]].astype(int)
            indexes_dist_classes[y_indices[k_ind]] = (distances[row_number, y_indices[k_ind]], class_recognized)
            counter += class_recognized

        # pairs handling
        if counter < k / 2:
            prediction = 0
        elif counter > k / 2:
            prediction = 1
        else:
            sum0 = 0
            sum1 = 0
            for value in indexes_dist_classes.values():
                if value[1] == 0:
                    sum0 += value[0]
                else:
                    sum1 += value[0]
            if sum0 > sum1:
                prediction = 0
            else:
                prediction = 1

        return prediction


class ArgumentParser:

    def __init__(self):
        self.parser = argparse.ArgumentParser(description='KNN algorithm')
        self.parser.add_argument('--train', help="Train dataset file", required=False)
        self.parser.add_argument('--test', help="Test dataset file", required=False)
        self.parser.add_argument('-k', '--krange', help="K range", type=int, nargs=2)
        self.parser.add_argument('--data', help="Dataset to be splitted on train and test set", required=False)
        self.parser.add_argument('--split', help="Split ratio", type=float, required=False)
        self.parser.add_argument('--predict', help="Dataset to predict", required=False)
        self.parser.add_argument('--output', help="Output file for prediction", required=False)
        self.args = self.parser.parse_args()


if __name__ == "__main__":

    knn_parser = ArgumentParser()
    file_manager = FileLoader(knn_parser.args.data)
    file_manager.train_and_test_split(knn_parser.args.split)
    file_manager.load_prediction(knn_parser.args.predict)

    #dict to store k and errors
    k_error = {}
    KNN = KNearestNeighbor()
    KNN.train(file_manager.train_features(), file_manager.train_labels())

    for k_param in range(knn_parser.args.krange[0], knn_parser.args.krange[1]):
        y_predicted = KNN.predict(file_manager.test_features(), k_param)
        error=0
        for label_number in range(len(y_predicted)):
            error += abs(y_predicted[label_number] - file_manager.test_labels()[label_number]) / file_manager.test_labels().shape[0]

        k_error[k_param] = error
        print("error:" + str(error) + " for k: " + str(k_param))

    if file_manager.prediction_set is not None:
        k_best = min(k_error, key=k_error.get)

        y_predicted = np.array(KNN.predict(file_manager.prediction_set, k_best))
        #nie wiem czy potrzebne
        #y_predicted = y_predicted.reshape((len(y_predicted), 1))
        output = np.concatenate((file_manager.prediction_set, y_predicted), axis=1)

        if knn_parser.args.output:
            np.savetxt(knn_parser.args.output, output, delimiter=' ')
        else:
            print(output)
