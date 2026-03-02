import numpy as np
from sklearn.neighbors import KNeighborsRegressor

def main():
    N = int(input("Enter N (positive integer): "))
    k = int(input("Enter k (positive integer): "))

    if N <= 0 or k <= 0:
        print("Error: N and k must be positive integers.")
        return

    X_train = np.zeros((N, 1), dtype=float)
    y_train = np.zeros(N, dtype=float)

    for i in range(N):
        X_train[i, 0] = float(input(f"Enter x for point {i+1}: "))
        y_train[i] = float(input(f"Enter y for point {i+1}: "))

    variance = np.var(y_train)

    x_query = float(input("Enter query X: "))

    if k > N:
        print("Error: k cannot be greater than N.")
        print("Variance of labels:", variance)
        return

    model = KNeighborsRegressor(n_neighbors=k, metric="euclidean")
    model.fit(X_train, y_train)

    y_pred = model.predict(np.array([[x_query]]))[0]

    print("Predicted Y:", y_pred)
    print("Variance of labels:", variance)


if __name__ == "__main__":
    main()