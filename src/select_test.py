
import numpy as np
from pathlib import Path
from scipy.io import loadmat

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw" / "cwru"
OUTPUT_DIR = PROJECT_ROOT / "pico_blink_test" / "model"

# INT8 quantisation parameters from the trained model
SCALE = 0.025995992124080658
ZERO_POINT = -3

# Available CWRU recordings
TESTS = {
    "1": ("97.mat", "Healthy", 0),
    "2": ("105.mat", "Inner Race", 1),
    "3": ("118.mat", "Ball Fault", 2),
    "4": ("130.mat", "Outer Race (6 o'clock)", 3),
    "5": ("144.mat", "Outer Race (3 o'clock)", 3),
    "6": ("156.mat", "Outer Race (12 o'clock)", 3),
}


def main():

    print("\n===================================")
    print("     BEARING FAULT TEST SELECTOR")
    print("===================================\n")

    for key, (filename, name, class_id) in TESTS.items():
        print(f"{key}. {name} - {filename}")

    choice = input("\nSelect a test (1-6): ").strip()

    if choice not in TESTS:
        print("Invalid selection.")
        return

    filename, name, class_id = TESTS[choice]

    mat_path = DATA_DIR / filename

    if not mat_path.exists():
        print(f"ERROR: File not found: {mat_path}")
        return

    print(f"\nLoading: {filename}")

    # Load MATLAB recording
    mat = loadmat(mat_path)

    # Find drive-end vibration signal
    signal_key = next(
        (key for key in mat if key.endswith("_DE_time")),
        None
    )

    if signal_key is None:
        print("ERROR: DE_time signal not found.")
        return

    signal = np.asarray(mat[signal_key]).flatten()

    if len(signal) < 1024:
        print("ERROR: Recording has fewer than 1024 samples.")
        return

    # Select the first 1024 samples
    sample = signal[:1024].astype(np.float32)

    # Apply model input quantisation
    quantized = np.clip(
        np.round(sample / SCALE) + ZERO_POINT,
        -128,
        127
    ).astype(np.int8)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output = OUTPUT_DIR / "test_input.h"

    # Generate embedded C header
    with output.open("w") as f:

        f.write("#pragma once\n")
        f.write("#include <cstdint>\n\n")

        f.write(f'#define TEST_NAME "{name}"\n')
        f.write(f"#define EXPECTED_CLASS {class_id}\n\n")

        f.write("const int8_t test_input[1024] = {\n")

        for i in range(0, 1024, 16):
            values = ", ".join(
                str(int(v)) for v in quantized[i:i+16]
            )
            f.write("    " + values + ",\n")

        f.write("};\n")

    print("\nTest input generated successfully!")
    print("-----------------------------------")
    print("Recording:", filename)
    print("Condition:", name)
    print("Expected class:", class_id)
    print("Samples:", len(quantized))
    print("Quantisation: INT8")
    print("Output:", output)
    print("-----------------------------------")
    print("\nNext: Rebuild and flash Pico 2.")


if __name__ == "__main__":
    main()
