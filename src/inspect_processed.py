
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"

X = np.load(DATA_DIR / "X.npy")
y = np.load(DATA_DIR / "y.npy")

CLASS_NAMES = [
    "Healthy",
    "Inner Race Fault",
    "Ball Fault",
    "Outer Race Fault"
]

print("X shape:", X.shape)
print("y shape:", y.shape)

print("\nFirst vibration window:")
print(X[0][:10])

print("\nFirst 10 labels:")
print(y[:10])

print("\nFirst window condition:")
print(CLASS_NAMES[y[0]])

print("\nUnique labels:")
print(np.unique(y))
