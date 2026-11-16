
import numpy as np
from pathlib import Path

# Load the processed CWRU dataset
X = np.load("../data/processed/X.npy")
y = np.load("../data/processed/y.npy")

# Select the first Healthy vibration window
class_id = 0
idx = int(np.where(y == class_id)[0][0])

sample = X[idx]

# INT8 quantization parameters from our model
scale = 0.025995992124080658
zero_point = -3

# Quantize the vibration samples
quantized = np.clip(
    np.round(sample / scale) + zero_point,
    -128,
    127
).astype(np.int8)

# Generate a C header
output = Path("model/test_input.h")

with output.open("w") as f:
    f.write("#pragma once\n")
    f.write("#include <cstdint>\n\n")
    f.write("const int8_t test_input[1024] = {\n")

    for i in range(0, 1024, 16):
        values = ", ".join(
            str(int(v)) for v in quantized[i:i+16]
        )
        f.write("    " + values + ",\n")

    f.write("};\n")

print("Test input generated successfully!")
print("Expected class: Healthy")
print("Samples:", len(quantized))
print("Output:", output)
