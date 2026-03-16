import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


class KNNClassifierSearch:
    def __init__(self, n_train: int, n_test: int):
        self.n_train = n_train
        self.n_test = n_test

        self.x_train = np.zeros((n_train, 1), dtype=float)
        self.y_train = np.zeros(n_train, dtype=int)

        self.x_test = np.zeros((n_test, 1), dtype=float)
        self.y_test = np.zeros(n_test, dtype=int)

    def insert_train_point(self, index: int, x_value: float, y_value: int) -> None:
        self.x_train[index, 0] = x_value
        self.y_train[index] = y_value

    def insert_test_point(self, index: int, x_value: float, y_value: int) -> None:
        self.x_test[index, 0] = x_value
        self.y_test[index] = y_value

    def find_best_k(self, k_min: int = 1, k_max: int = 10):
        best_k = 1
        best_accuracy = -1.0

        upper_k = min(k_max, self.n_train)

        for k in range(k_min, upper_k + 1):
            model = KNeighborsClassifier(n_neighbors=k)
            model.fit(self.x_train, self.y_train)

            y_pred = model.predict(self.x_test)
            accuracy = accuracy_score(self.y_test, y_pred)

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_k = k

        return best_k, best_accuracy


def read_positive_integer(prompt: str) -> int:
    while True:
        try:
            value = int(input(prompt))
            if value > 0:
                return value
            print("Please enter a positive integer.")
        except ValueError:
            print("Invalid input. Please enter a positive integer.")


def read_non_negative_integer(prompt: str) -> int:
    while True:
        try:
            value = int(input(prompt))
            if value >= 0:
                return value
            print("Please enter a non-negative integer.")
        except ValueError:
            print("Invalid input. Please enter a non-negative integer.")


def read_float(prompt: str) -> float:
    while True:
        try:
            return float(input(prompt))
        except ValueError:
            print("Invalid input. Please enter a real number.")


def main():
    n = read_positive_integer("Enter N (number of training pairs): ")

    classifier_search = None

    x_train_temp = np.zeros((n, 1), dtype=float)
    y_train_temp = np.zeros(n, dtype=int)

    print("\nEnter training set pairs:")
    for i in range(n):
        x = read_float(f"Train pair {i + 1} - enter x: ")
        y = read_non_negative_integer(f"Train pair {i + 1} - enter y: ")
        x_train_temp[i, 0] = x
        y_train_temp[i] = y

    m = read_positive_integer("\nEnter M (number of test pairs): ")

    classifier_search = KNNClassifierSearch(n, m)

    classifier_search.x_train = x_train_temp
    classifier_search.y_train = y_train_temp

    print("\nEnter test set pairs:")
    for i in range(m):
        x = read_float(f"Test pair {i + 1} - enter x: ")
        y = read_non_negative_integer(f"Test pair {i + 1} - enter y: ")
        classifier_search.insert_test_point(i, x, y)

    best_k, best_accuracy = classifier_search.find_best_k(1, 10)

    print(f"\nBest k: {best_k}")
    print(f"Test accuracy: {best_accuracy:.4f}")


if __name__ == "__main__":
    main()