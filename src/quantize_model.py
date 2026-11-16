
from pathlib import Path

import numpy as np
import tensorflow as tf

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = PROJECT_ROOT / "models" / "bearing_cnn.keras"
DATA_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_PATH = PROJECT_ROOT / "models" / "bearing_cnn_int8.tflite"

NUM_CALIBRATION_SAMPLES_PER_CLASS = 50


def main():

    print("Loading trained CNN...")

    model = tf.keras.models.load_model(MODEL_PATH)

    X = np.load(DATA_DIR / "X.npy")
    y = np.load(DATA_DIR / "y.npy")

    X = X[..., np.newaxis].astype(np.float32)

    print("Dataset shape:", X.shape)

    # Select representative samples from every class.
    rng = np.random.default_rng(42)
    calibration_indices = []

    for class_id in range(4):

        class_indices = np.where(y == class_id)[0]

        selected = rng.choice(
            class_indices,
            size=min(
                NUM_CALIBRATION_SAMPLES_PER_CLASS,
                len(class_indices)
            ),
            replace=False
        )

        calibration_indices.extend(selected)

        print(
            f"Class {class_id}: "
            f"{len(selected)} calibration samples"
        )

    rng.shuffle(calibration_indices)

    calibration_X = X[calibration_indices]

    print(
        "\nTotal calibration samples:",
        len(calibration_X)
    )

    def representative_dataset():

        for sample in calibration_X:
            yield [sample[np.newaxis, ...]]

    print("\nConverting model to INT8...")

    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    converter.optimizations = [
        tf.lite.Optimize.DEFAULT
    ]

    converter.representative_dataset = representative_dataset

    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS_INT8
    ]

    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_bytes(tflite_model)

    original_size = MODEL_PATH.stat().st_size
    quantized_size = OUTPUT_PATH.stat().st_size

    print("\nQuantisation complete!")

    print(f"Original Keras file: {original_size:,} bytes")
    print(f"INT8 TFLite file: {quantized_size:,} bytes")

    print(
        "Size reduction:",
        f"{(1 - quantized_size / original_size) * 100:.2f}%"
    )

    print("\nSaved:", OUTPUT_PATH)

    interpreter = tf.lite.Interpreter(
        model_path=str(OUTPUT_PATH)
    )

    interpreter.allocate_tensors()

    print("\nInput details:")
    print(interpreter.get_input_details())

    print("\nOutput details:")
    print(interpreter.get_output_details())


if __name__ == "__main__":
    main()
