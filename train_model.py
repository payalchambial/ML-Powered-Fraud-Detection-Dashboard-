
from pathlib import Path
import json
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    confusion_matrix,
    classification_report
)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "creditcard.csv"
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"

MODEL_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

def main():
    if not DATA_PATH.exists():
        print("Dataset not found!")
        print("Download creditcard.csv from Kaggle.")
        print(f"Save it inside: {DATA_PATH}")
        return

    df = pd.read_csv(DATA_PATH)

    if "Class" not in df.columns:
        raise ValueError("Dataset must contain a 'Class' column.")

    df = df.dropna(subset=["Class"]).copy()
    df["Class"] = df["Class"].astype(int)

    X = df.drop(columns=["Class"])
    y = df["Class"]

    if y.nunique() != 2:
        raise ValueError("Dataset must contain both classes: 0 and 1.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42
        ))
    ])

    print("Training AI Fraud Detection model...")
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    metrics = {
        "total_transactions": int(len(df)),
        "fraud_cases": int((y == 1).sum()),
        "normal_cases": int((y == 0).sum()),
        "precision": float(
            precision_score(y_test, predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(y_test, predictions, zero_division=0)
        ),
        "f1_score": float(
            f1_score(y_test, predictions, zero_division=0)
        ),
        "average_precision": float(
            average_precision_score(y_test, probabilities)
        ),
        "confusion_matrix": confusion_matrix(
            y_test, predictions
        ).tolist(),
        "features": list(X.columns),
        "classification_report": classification_report(
            y_test,
            predictions,
            target_names=["Normal", "Fraud"],
            output_dict=True,
            zero_division=0
        )
    }

    joblib.dump(model, MODEL_DIR / "fraud_model.joblib")

    with open(OUTPUT_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("\nModel Evaluation")
    print(classification_report(
        y_test,
        predictions,
        target_names=["Normal", "Fraud"],
        zero_division=0
    ))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))
    print("\nModel saved successfully!")
    print("Now run: streamlit run app.py")

if __name__ == "__main__":
    main()
