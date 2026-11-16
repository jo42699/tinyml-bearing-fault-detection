
from pathlib import Path

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

# --------------------------------------------------
# 1. Project configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data" / "raw" / "cwru"
OUTPUT_DIR = PROJECT_ROOT / "results" / "plots"

DATA_FILES = {
    "97.mat": "Healthy",
    "105.mat": "Inner Race Fault",
    "118.mat": "Ball Fault",
    "130.mat": "Outer Race Fault - 6 o'clock",
    "144.mat": "Outer Race Fault - 3 o'clock",
    "156.mat": "Outer Race Fault - 12 o'clock",
}


def load_signal(file_path):

    mat_data = sio.loadmat(file_path)

    signal_key = next(
        key for key in mat_data
        if key.endswith("_DE_time")
    )

    return np.asarray(
        mat_data[signal_key],
        dtype=np.float64
    ).flatten()



# 2. Load vibration data
# --------------------------------------------------

def load_vibration_data(file_path):
    """Load drive-end vibration measurements from a MAT file."""

    mat_data = sio.loadmat(file_path)

    # Find the drive-end vibration variable
    signal_keys = [
        key for key in mat_data
        if key.endswith("_DE_time")
    ]

    if not signal_keys:
        raise ValueError(
            f"No drive-end vibration signal found in {file_path.name}"
        )

    signal = np.asarray(
        mat_data[signal_keys[0]],
        dtype=np.float64
    ).flatten()

    return signal, signal_keys[0]


# --------------------------------------------------
# 3. Calculate signal statistics
# --------------------------------------------------

def calculate_statistics(signal):
    """Calculate basic statistical measurements."""

    statistics = {
        "samples": len(signal),
        "mean": np.mean(signal),
        "std": np.std(signal),
        "minimum": np.min(signal),
        "maximum": np.max(signal),
        "rms": np.sqrt(np.mean(signal ** 2)),
    }

    return statistics


# --------------------------------------------------
# 4. Main program
# --------------------------------------------------

def main():

    print("=" * 55)
    print("ARM TINYML - CWRU DATASET EXPLORER")
    print("=" * 55)

    print(f"\nDataset location: {DATA_DIR}")

    for filename, condition in DATA_FILES.items():

        file_path = DATA_DIR / filename

        print("\n" + "-" * 55)
        print(f"File: {filename}")
        print(f"Condition: {condition}")

        if not file_path.exists():
            print("ERROR: File not found!")
            continue

        # Load the vibration signal
        signal, variable_name = load_vibration_data(file_path)

        # Calculate statistics
        stats = calculate_statistics(signal)

        print(f"MATLAB variable: {variable_name}")
        print(f"Number of samples: {stats['samples']:,}")
        print(f"Mean: {stats['mean']:.6f}")
        print(f"Standard deviation: {stats['std']:.6f}")
        print(f"Minimum: {stats['minimum']:.6f}")
        print(f"Maximum: {stats['maximum']:.6f}")
        print(f"RMS: {stats['rms']:.6f}")

    print("\n" + "=" * 55)
    print("DATASET INSPECTION COMPLETE")
    print("=" * 55)


if __name__ == "__main__":
    main()
