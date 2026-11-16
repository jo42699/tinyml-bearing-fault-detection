
from pathlib import Path

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "raw" / "cwru"
OUTPUT_DIR = PROJECT_ROOT / "results" / "plots"

DATA_FILES = {
    "97.mat": "Healthy",
    "105.mat": "Inner Race Fault",
    "118.mat": "Ball Fault",
    "130.mat": "Outer Race - 6 o'clock",
    "144.mat": "Outer Race - 3 o'clock",
    "156.mat": "Outer Race - 12 o'clock",
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


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(
        6, 1,
        figsize=(12, 14)
    )

    for ax, (filename, condition) in zip(
        axes, DATA_FILES.items()
    ):
        signal = load_signal(DATA_DIR / filename)

        # Plot the first 2,000 samples
        samples = signal[:2000]

        ax.plot(samples, linewidth=0.6)
        ax.set_title(condition)
        ax.set_xlabel("Sample number")
        ax.set_ylabel("Amplitude")
        ax.grid(True)

    plt.tight_layout()

    output_file = OUTPUT_DIR / "vibration_signals.png"

    plt.savefig(output_file, dpi=150)

    print(f"Plot saved to: {output_file}")

    plt.show()


if __name__ == "__main__":
    main()
