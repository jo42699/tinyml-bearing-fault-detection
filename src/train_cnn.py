
from pathlib import Path
from xml.parsers.expat import model

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "results" / "plots"

CLASS_NAMES = [
    "Healthy",
    "Inner Race Fault",
    "Ball Fault",
    "Outer Race Fault"
]

BATCH_SIZE = 32
EPOCHS = 30


def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load the processed dataset
    X = np.load(DATA_DIR / "X.npy")
    y = np.load(DATA_DIR / "y.npy")

    print("Original X shape:", X.shape)
    print("Original y shape:", y.shape)

    # CNN expects (samples, timesteps, channels)
    X = X[..., np.newaxis]

    print("CNN input shape:", X.shape)

    # Initial development split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Split some training data for validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.2,
        random_state=42,
        stratify=y_train
    )

    print("Training:", X_train.shape)
    print("Validation:", X_val.shape)
    print("Testing:", X_test.shape)

    # Build the 1D CNN
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(1024, 1)),

        tf.keras.layers.Conv1D(
            filters=16,
            kernel_size=5,
            activation="relu"
        ),

        tf.keras.layers.MaxPooling1D(pool_size=2),

        tf.keras.layers.Conv1D(
            filters=32,
            kernel_size=5,
            activation="relu"
        ),

        tf.keras.layers.MaxPooling1D(pool_size=2),

        tf.keras.layers.Conv1D(
            filters=64,
            kernel_size=3,
            activation="relu"
        ),

        tf.keras.layers.GlobalAveragePooling1D(),

        tf.keras.layers.Dense(32, activation="relu"),

        tf.keras.layers.Dense(
            4,
            activation="softmax"
        )
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    print("\nModel architecture:")
    model.summary()

    # Stop training if validation loss stops improving
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    )

    print("\nTraining CNN...")

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stopping],
        verbose=1
    )

    # Evaluate
    print("\nEvaluating CNN...")

    predictions = model.predict(X_test)
    predicted_classes = np.argmax(predictions, axis=1)

    accuracy = accuracy_score(y_test, predicted_classes)

    print(f"\nCNN accuracy: {accuracy:.4f}")

    print("\nClassification report:")

    print(
        classification_report(
            y_test,
            predicted_classes,
            labels=[0, 1, 2, 3],
            target_names=CLASS_NAMES,
            zero_division=0
        )
    )

    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, predicted_classes))

    # Save model
    model_dir = PROJECT_ROOT / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    model.save(model_dir / "bearing_cnn.keras")

    # Plot training history
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history.history["accuracy"], label="Training")
    axes[0].plot(history.history["val_accuracy"], label="Validation")
    axes[0].set_title("CNN Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(history.history["loss"], label="Training")
    axes[1].plot(history.history["val_loss"], label="Validation")
    axes[1].set_title("CNN Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()

    plot_path = OUTPUT_DIR / "cnn_training_history.png"
    plt.savefig(plot_path, dpi=150)

    print("\nSaved model and training plot.")
    print("Model:", PROJECT_ROOT / "models" / "bearing_cnn.keras")
    print("Plot:", plot_path)

    plt.show()


if __name__ == "__main__":
    main()
