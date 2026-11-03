import json
import os

import joblib
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


def train_model(n_estimators=100, max_depth=5, random_state=42):
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
    )
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    return model, accuracy


if __name__ == "__main__":
    model, accuracy = train_model()
    joblib.dump(model, "model.pkl")
    os.makedirs("metrics", exist_ok=True)
    with open("metrics/metrics.json", "w") as f:
        json.dump({"accuracy": round(accuracy, 4)}, f, indent=2)
    print(f"Model trained. Accuracy: {accuracy:.4f}")