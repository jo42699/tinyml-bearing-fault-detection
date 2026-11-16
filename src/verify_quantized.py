
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"

CLASS_NAMES = [
    "Healthy",
    "Inner Race Fault",
    "Ball Fault",
    "Outer Race Fault"
]


def main():

    # Load data
    X = np.load(DATA_DIR / "X.npy")
    y = np.load(DATA_DIR / "y.npy")

    X = X[..., np.newaxis].astype(np.float32)

    # Load original floating-point model
    print("Loading FP32 model...")

    float_model = tf.keras.models.load_model(
        MODEL_DIR / "bearing_cnn.keras"
    )

    # Get FP32 predictions
    print("Running FP32 predictions...")

    float_predictions = float_model.predict(
        X,
        batch_size=32,
        verbose=0
    )

    float_classes = np.argmax(float_predictions, axis=1)

    # Load INT8 model
    print("Loading INT8 model...")

    interpreter = tf.lite.Interpreter(
        model_path=str(MODEL_DIR / "bearing_cnn_int8.tflite")
    )

    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    input_scale, input_zero_point = input_details["quantization"]
    output_scale, output_zero_point = output_details["quantization"]

    print("\nInput quantisation:")
    print("Scale:", input_scale)
    print("Zero point:", input_zero_point)

    print("\nOutput quantisation:")
    print("Scale:", output_scale)
    print("Zero point:", output_zero_point)

    # Run INT8 inference
    int8_predictions = []

    print("\nRunning INT8 predictions...")

    for window in X:

        # Convert float input to INT8
        quantized_input = np.round(
            window / input_scale + input_zero_point
        )

        quantized_input = np.clip(
            quantized_input,
            -128,
            127
        ).astype(np.int8)

        interpreter.set_tensor(
            input_details["index"],
            quantized_input[np.newaxis, ...]
        )

        interpreter.invoke()

        output = interpreter.get_tensor(
            output_details["index"]
        )

        # Convert INT8 output back to float
        dequantized_output = (
            output.astype(np.float32) - output_zero_point
        ) * output_scale

        int8_predictions.append(dequantized_output[0])

    int8_predictions = np.array(int8_predictions)

    int8_classes = np.argmax(int8_predictions, axis=1)

    # Compare results
    float_accuracy = accuracy_score(y, float_classes)
    int8_accuracy = accuracy_score(y, int8_classes)

    agreement = np.mean(
        float_classes == int8_classes
    )

    print("\n" + "=" * 45)
    print("FP32 vs INT8 COMPARISON")
    print("=" * 45)

    print(f"FP32 accuracy:       {float_accuracy:.4f}")
    print(f"INT8 accuracy:       {int8_accuracy:.4f}")
    print(f"Prediction agreement: {agreement * 100:.2f}%")

    print("\nINT8 class predictions:")

    for label, name in enumerate(CLASS_NAMES):
        count = np.sum(int8_classes == label)
        print(f"{name}: {count}")

    print("\nVerification complete!")


if __name__ == "__main__":
    main()
