from pathlib import Path

import tensorflow as tf

model_path = Path("models/bearing_cnn_int8.tflite")
output_path = Path("pico_blink_test/model/bearing_model.h")

model_data = model_path.read_bytes()

with output_path.open("w") as f:
    f.write("#pragma once\n")
    f.write("#include <cstdint>\n\n")
    f.write("alignas(16) const unsigned char bearing_model[] = {\n")

    for i, byte in enumerate(model_data):
        if i % 12 == 0:
            f.write("    ")

        f.write(f"0x{byte:02x}, ")

        if i % 12 == 11:
            f.write("\n")

    f.write("\n};\n")
    f.write(f"const unsigned int bearing_model_len = {len(model_data)};\n")

print(f"Created {output_path}")
print(f"Model size: {len(model_data):,} bytes")