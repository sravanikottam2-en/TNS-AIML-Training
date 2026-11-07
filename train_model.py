"""Train a demo supervised classifier on synthetic, non-clinical data.

The generated labels are simulated for software demonstration only. This model
must not be used to estimate a real person's medical risk.
"""
import os
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

def train_and_save(path="diabetes_model.joblib"):
    rng = np.random.default_rng(42)
    n = 6000
    pregnancies = rng.integers(0, 14, n)
    glucose = np.clip(rng.normal(120, 30, n), 55, 220)
    blood_pressure = np.clip(rng.normal(72, 12, n), 35, 125)
    skin = np.clip(rng.normal(25, 10, n), 5, 70)
    insulin = np.clip(rng.lognormal(4.4, 0.55, n), 10, 400)
    bmi = np.clip(rng.normal(30, 7, n), 15, 55)
    pedigree = np.clip(rng.gamma(2, 0.25, n), 0.05, 2.5)
    age = rng.integers(18,  eighty := 81, n)
    # Simulated outcome relationship; not a medical risk equation.
    z = (-5.0 + 0.035*(glucose-100) + 0.055*(bmi-25)
         + 0.018*(age-35) + 0.13*pregnancies + 0.45*(pedigree-0.5))
    p = 1 / (1 + np.exp(-z))
    y = rng.binomial(1, p)
    X = np.column_stack([pregnancies, glucose, blood_pressure, skin,
                         insulin, bmi, pedigree, age])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    model = RandomForestClassifier(n_estimators=250, max_depth=8,
                                   min_samples_leaf=5, class_weight="balanced",
                                   random_state=42)
    model.fit(X_train, y_train)
    print("Demo model accuracy:", round(accuracy_score(y_test, model.predict(X_test)), 3))
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    joblib.dump(model, path)
    return model

if __name__ == "__main__":
    train_and_save()
