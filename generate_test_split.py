import numpy as np

# Dummy test data — adjust shape and values as needed
X_test = np.random.rand(10, 4096)  # Example: 10 samples, 64x64 flattened images
y_test = np.random.randint(0, 40, size=10)  # Example: 40 classes

np.savez("test_split.npz", X_test=X_test, y_test=y_test)
print("test_split.npz generated successfully.")
