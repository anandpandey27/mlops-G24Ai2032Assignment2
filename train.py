# train.py
# Trains a DecisionTreeClassifier on sklearn's Olivetti faces dataset
# Saves model as 'savedmodel.pth' (as required by assignment)

from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import joblib
import numpy as np

def main():
    data = fetch_olivetti_faces()
    X = data.images  # shape (400, 64, 64)
    y = data.target  # shape (400,)

    # Flatten images to 1D vectors (required for scikit-learn classifiers)
    X_flat = X.reshape((X.shape[0], -1))

    # 70% train, 30% test
    X_train, X_test, y_train, y_test = train_test_split(
        X_flat, y, test_size=0.30, random_state=42, stratify=y
    )

    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X_train, y_train)

    # save model to savedmodel.pth (assignment requires this filename)
    joblib.dump(clf, "savedmodel.pth")

    # Optionally print training accuracy
    train_acc = accuracy_score(y_train, clf.predict(X_train))
    test_acc = accuracy_score(y_test, clf.predict(X_test))
    print(f"Training accuracy: {train_acc:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")

    # Save the dataset split indices so test.py can reproduce the test split deterministically
    np.savez("test_data.npz", x=X_test, y=y_test)
