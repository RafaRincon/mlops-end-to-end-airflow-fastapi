#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train multiple classifiers, select the best by CV (macro-F1) and log everything to MLflow.
This version logs artifacts (figures/reports) **directly** to MLflow using mlflow.log_figure()
and mlflow.log_text(), so PNG/TXT files appear only in the run's artifact store.
"""

import os
import json
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

import matplotlib.pyplot as plt

import mlflow
import mlflow.sklearn

TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
mlflow.set_tracking_uri(TRACKING_URI)
print("Tracking URI:", mlflow.get_tracking_uri())

# ------------------------
# CLI
# ------------------------
parser = argparse.ArgumentParser(description="Train and log best model to MLflow (direct artifact logging).")
parser.add_argument("--data", type=str, default="./data/Synthetic_Customers_Data.csv", help="Path to CSV dataset.")
parser.add_argument("--experiment", type=str, default="customer_quality", help="MLflow experiment name.")
parser.add_argument("--register", type=str, default="", help="Optional model registry name to register the best model.")
parser.add_argument("--test_size", type=float, default=0.2, help="Test size for train/test split.")
parser.add_argument("--cv_folds", type=int, default=5, help="Number of CV folds.")
parser.add_argument("--alias", type=str, default="Production", help="Alias of the model.")

args = parser.parse_args()

# ------------------------
# Data
# ------------------------
df = pd.read_csv(args.data)
X = df.drop(columns=["customer_type"])
y = df["customer_type"]

classes = np.unique(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=args.test_size, stratify=y, random_state=42
)

# ------------------------
# Models & grids
# ------------------------
models_and_grids = {
    "logreg": (
        Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=200, multi_class="auto"))
        ]),
        {"clf__C": [0.1, 1.0, 10.0]}
    ),
    "rf": (
        Pipeline([("clf", RandomForestClassifier(random_state=42))]),
        {
            "clf__n_estimators": [150, 300],
            "clf__max_depth": [None, 5, 10],
            "clf__min_samples_split": [2, 5],
        },
    ),
    "gb": (
        Pipeline([("clf", GradientBoostingClassifier(random_state=42))]),
        {
            "clf__n_estimators": [100, 200],
            "clf__learning_rate": [0.05, 0.1],
            "clf__max_depth": [2, 3],
        },
    ),
}

scoring = {"f1_macro": "f1_macro", "accuracy": "accuracy"}
cv = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=42)

# Ensure experiment exists
mlflow.set_experiment(args.experiment)

best_overall = {"model_key": None, "cv_f1_macro": -1.0, "run_id": None, "best_params": None}

# ------------------------
# Train + log each model
# ------------------------
for model_key, (pipe, grid) in models_and_grids.items():
    with mlflow.start_run(run_name=model_key) as run:
        print("RUN:", run.info.run_id)
        print("ARTIFACT_URI:", mlflow.get_artifact_uri())

        run_id = run.info.run_id

        mlflow.sklearn.autolog(log_datasets=False, silent=True)

        gs = GridSearchCV(
            estimator=pipe,
            param_grid=grid,
            scoring=scoring,
            refit="f1_macro",
            cv=cv,
            n_jobs=-1,
            verbose=0,
        )
        gs.fit(X_train, y_train)

        # CV metrics
        mean_f1 = gs.cv_results_["mean_test_f1_macro"][gs.best_index_]
        mean_acc = gs.cv_results_["mean_test_accuracy"][gs.best_index_]
        mlflow.log_metric("cv_mean_f1_macro", float(mean_f1))
        mlflow.log_metric("cv_mean_accuracy", float(mean_acc))
        mlflow.log_params(gs.best_params_)

        # Test eval
        y_pred = gs.best_estimator_.predict(X_test)
        test_f1 = f1_score(y_test, y_pred, average="macro")
        test_acc = accuracy_score(y_test, y_pred)
        mlflow.log_metric("test_f1_macro", float(test_f1))
        mlflow.log_metric("test_accuracy", float(test_acc))

        # (Optional) Multiclass ROC-AUC
        try:
            if hasattr(gs.best_estimator_, "predict_proba"):
                proba = gs.best_estimator_.predict_proba(X_test)
                from sklearn.preprocessing import label_binarize
                Y_test_bin = label_binarize(y_test, classes=classes)
                if isinstance(proba, list):
                    proba = np.vstack([p[:, 1] for p in proba]).T
                roc_ovr = roc_auc_score(Y_test_bin, proba, multi_class="ovr")
                mlflow.log_metric("test_roc_auc_ovr", float(roc_ovr))
        except Exception as e:
            mlflow.log_param("roc_auc_warning", f"{type(e).__name__}: {e}")

        # Log report directly (no local file)
        report = classification_report(y_test, y_pred, output_dict=False)
        mlflow.log_text(report, f"reports/{model_key}_classification_report.txt")

        # Confusion matrix figure -> log_figure (goes straight to artifacts)
        cm = confusion_matrix(y_test, y_pred, labels=classes)
        fig = plt.figure()
        plt.imshow(cm, interpolation="nearest")
        plt.title(f"Confusion Matrix - {model_key}")
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)
        plt.xlabel("Predicted label")
        plt.ylabel("True label")
        plt.tight_layout()
        mlflow.log_figure(fig, f"figures/{model_key}_confusion_matrix.png")
        plt.close(fig)

        # Log the trained model
        mlflow.sklearn.log_model(
            sk_model=gs.best_estimator_,
            artifact_path="model",
            input_example=X_train.iloc[:5],
            registered_model_name=None,
        )

        if mean_f1 > best_overall["cv_f1_macro"]:
            best_overall.update(
                {
                    "model_key": model_key,
                    "cv_f1_macro": float(mean_f1),
                    "run_id": run_id,
                    "best_params": gs.best_params_,
                }
            )

# ------------------------
# Tag the best run, optionally register it
# ------------------------
if best_overall["run_id"]:
    client = mlflow.tracking.MlflowClient()
    client.set_tag(best_overall["run_id"], "is_best", "true")
    client.set_tag(best_overall["run_id"], "best_model_key", best_overall["model_key"])
    client.set_tag(best_overall["run_id"], "cv_best_f1_macro", str(best_overall["cv_f1_macro"]))

    if args.register:
        # 1) Register the best run's model
        best_model_uri = f"runs:/{best_overall['run_id']}/model"
        try:
            mv = mlflow.register_model(best_model_uri, args.register)
            print(f"Registered best model as '{args.register}': version {mv.version}")

            # 2) Set alias to point to this version (this is your "production" pointer)
            #    Use 'champion' as the alias your FastAPI will load.
            client.set_registered_model_alias(
                name=args.register,
                alias=args.alias,
                version=mv.version
            )


        except Exception as e:
            print(f"Model registry error: {e}")

# ------------------------
# Persist summary (also log to artifacts)
# ------------------------
summary = {
    "best_model_key": best_overall["model_key"],
    "best_cv_f1_macro": best_overall["cv_f1_macro"],
    "best_run_id": best_overall["run_id"],
    "best_params": best_overall["best_params"],
    "experiment": args.experiment,
}
mlflow.log_text(json.dumps(summary, indent=2), "best_summary.json")
print(json.dumps(summary, indent=2))