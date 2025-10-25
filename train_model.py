import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib

DATA_CSV = "sign_data.csv"
MODEL_FILE = "sign_model.joblib"

def load_data():
    df = pd.read_csv(DATA_CSV)
    X = df.drop(columns=["label"]).values
    y = df["label"].values
    return X, y

def main():
    X, y = load_data()

    # --- Dynamically adjust test_size ---
    num_classes = len(np.unique(y))
    total_samples = len(y)

    if total_samples < num_classes * 2:
        print(f"[WARN] Very few samples ({total_samples}) for {num_classes} classes.")
        print("Try collecting more data for better accuracy!")

    # ensure at least one test sample per class
    min_test_size = num_classes / total_samples + 0.05
    test_size = max(0.2, min_test_size)
    test_size = min(test_size, 0.5)

    print(f"[INFO] Using test_size={test_size:.2f} for {num_classes} classes and {total_samples} samples.")

    # --- Split data safely ---
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, stratify=y, test_size=test_size, random_state=42
        )
    except ValueError:
        print("[WARN] Stratified split failed — falling back to random split.")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

    # --- Train model ---
    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    print("Training model...")
    clf.fit(X_train, y_train)

    # --- Evaluate ---
    preds = clf.predict(X_test)
    print("\n[RESULTS]")
    print("Accuracy:", accuracy_score(y_test, preds))
    print(classification_report(y_test, preds))

    # --- Save model ---
    joblib.dump(clf, MODEL_FILE)
    print(f"\n[INFO] Saved model to {MODEL_FILE}")

if __name__ == "__main__":
    main()
