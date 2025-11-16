# test.py
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Load dataset
data = fetch_olivetti_faces()
X, y = data.data, data.target

# Split data
_, X_test, _, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Load model
clf = joblib.load('savedmodel.pth')

# Evaluate
accuracy = clf.score(X_test, y_test)
print(f"Test Accuracy: {accuracy:.2f}")
