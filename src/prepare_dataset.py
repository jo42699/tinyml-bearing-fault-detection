
from pathlib import Path

import numpy as np
import scipy.io as sio

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "raw" / "cwru"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

WINDOW_SIZE = 1024
STEP_SIZE = 512

DATA_FILES = {
    "97.mat": 0,   # Healthy
    "105.mat": 1,  # Inner race
    "118.mat": 2,  # Ball fault
    "130.mat": 3,  # Outer race, 6 o'clock
    "144.mat": 3,  # Outer race, 3 o'clock
    "156.mat": 3,  # Outer race, 12 o'clock
}

CLASS_NAMES = [
    "Healthy",
    "Inner Race Fault",
    "Ball Fault",
    "Outer Race Fault",
]


def load_signal(file_path):
    mat_data = sio.loadmat(file_path)

    signal_key = next(
        key for key in mat_data
        if key.endswith("_DE_time")
    )

    return np.asarray(
        mat_data[signal_key],
        dtype=np.float32
    ).flatten()


def create_windows(signal, label):
    windows = []

    for start in range(
        0,
        len(signal) - WINDOW_SIZE + 1,
        STEP_SIZE
    ):
        window = signal[start:start + WINDOW_SIZE]

        windows.append(window)

    X = np.asarray(windows, dtype=np.float32)
    y = np.full(len(X), label, dtype=np.int64)

    return X, y


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_X = []
    all_y = []

    print("=" * 55)
    print("TINYML - DATASET PREPROCESSING")
    print("=" * 55)

    for filename, label in DATA_FILES.items():
        signal = load_signal(DATA_DIR / filename)

        X, y = create_windows(signal, label)

        all_X.append(X)
        all_y.append(y)

        print(
            f"{filename}: {len(X)} windows | "
            f"Class: {CLASS_NAMES[label]}"
        )

    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)

    np.save(OUTPUT_DIR / "X.npy", X)
    np.save(OUTPUT_DIR / "y.npy", y)

    print("-" * 55)
    print("Dataset preparation complete!")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")

    print("\nClass distribution:")

    for label, name in enumerate(CLASS_NAMES):
        count = np.sum(y == label)
        print(f"{name}: {count} windows")

    print("\nSaved:")
    print(OUTPUT_DIR / "X.npy")
    print(OUTPUT_DIR / "y.npy")


if __name__ == "__main__":
    main()

