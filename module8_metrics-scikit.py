import numpy as np
from sklearn.metrics import precision_score, recall_score

def read_positive_integer(prompt: str) -> int:
    while True:
        try:
            value = int(input(prompt))
            if value > 0:
                return value
            print("Please enter a positive integer.")
        except ValueError:
            print("Invalid input. Please enter an integer.")


def read_binary_value(prompt: str) -> int:
    while True:
        try:
            value = int(input(prompt))
            if value in (0, 1):
                return value
            print("Please enter only 0 or 1.")
        except ValueError:
            print("Invalid input. Please enter 0 or 1.")


def main() -> None:
    n = read_positive_integer("Enter N (positive integer): ")

    x_true = np.zeros(n, dtype=int)
    y_pred = np.zeros(n, dtype=int)

    for i in range(n):
        print(f"\nPoint {i + 1}:")
        x_true[i] = read_binary_value("  Enter x (ground truth, 0 or 1): ")
        y_pred[i] = read_binary_value("  Enter y (predicted, 0 or 1): ")

    precision = precision_score(x_true, y_pred, zero_division=0)
    recall = recall_score(x_true, y_pred, zero_division=0)

    print("\nResults:")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")


if __name__ == "__main__":
    main()