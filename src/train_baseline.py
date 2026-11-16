
from pathlib import Path

import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed"

CLASS_NAMES = [
    "Healthy",
    "Inner Race Fault",
    "Ball Fault",
    "Outer Race Fault"
]


def extract_features(window):
    features = [
        np.mean(window),
        np.std(window),
        np.min(window),
        np.max(window),
        np.sqrt(np.mean(window ** 2)),
        np.ptp(window),
        np.mean(np.abs(window)),
        np.var(window)
    ]

    return features


def main():
    X = np.load(DATA_DIR / "X.npy")
    y = np.load(DATA_DIR / "y.npy")

    print("Extracting features...")

    X_features = np.array([
        extract_features(window)
        for window in X
    ])

    print("Feature shape:", X_features.shape)

    # Stratified random split for an initial baseline only.
    # This is NOT an independent test because windows overlap
    # and originate from the same recordings.
    X_train, X_test, y_train, y_test = train_test_split(
        X_features,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("Training samples:", len(X_train))
    print("Test samples:", len(X_test))

    model = make_pipeline(
        StandardScaler(),
        RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight="balanced"
        )
    )

    print("\nTraining Random Forest...")
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    print("\nBaseline results")
    print("-------------------------")
    print(f"Accuracy: {accuracy:.4f}")

    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            predictions,
            labels=[0, 1, 2, 3],
            target_names=CLASS_NAMES,
            zero_division=0
        )
    )

    print("Confusion matrix:")
    print(confusion_matrix(y_test, predictions))


if __name__ == "__main__":
    main()
