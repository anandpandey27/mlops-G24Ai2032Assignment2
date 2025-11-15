# test.py
# Loads savedmodel.pth and computes accuracy on test set

import joblib
import numpy as np
from sklearn.metrics import accuracy_score

def main():
    clf = joblib.load("savedmodel.pth")
    arr = np.load("test_split.npz")
    X_test = arr["X_test"]
    y_test = arr["y_test"]

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {acc:.4f}")

if __name__ == "__main__":
    main()
