import numpy as np

class KNNRegressor:
    def __init__(self, x_values: np.ndarray, y_values: np.ndarray):
        self.x = x_values
        self.y = y_values

    def predict(self, x_query: float, k: int) -> float:
        distances = np.abs(self.x - x_query)

        nearest_indices = np.argsort(distances)[:k]

        return np.mean(self.y[nearest_indices])


def main():
    N = int(input("Enter N (positive integer): "))
    k = int(input("Enter k (positive integer): "))

    x_values = np.zeros(N, dtype=float)
    y_values = np.zeros(N, dtype=float)

    for i in range(N):
        x_values[i] = float(input(f"Enter x for point {i+1}: "))
        y_values[i] = float(input(f"Enter y for point {i+1}: "))

    x_query = float(input("Enter query X: "))

    if k > N:
        print("Error: k cannot be greater than N.")
        return 

    model = KNNRegressor(x_values, y_values)
    result = model.predict(x_query, k)

    print("Predicted Y:", result)


if __name__ == "__main__":
    main()