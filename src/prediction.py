"""Supervised travel-behavior prediction: classify traveler_type and compare models."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, precision_recall_fscore_support)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from src.config import (PLOTS_DIR, RANDOM_STATE, REPORTS_DIR, TEST_SIZE)
from src.eda import save_plot
from src.preprocessing import build_preprocessor


def get_models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=3000, C=1.0),
        "Random Forest": RandomForestClassifier(n_estimators=400, min_samples_leaf=2,
                                                random_state=RANDOM_STATE, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=300,
                                                        learning_rate=0.05,
                                                        max_depth=3,
                                                        random_state=RANDOM_STATE),
        "SVM (RBF)": SVC(C=3.0, gamma="scale", probability=True,
                         random_state=RANDOM_STATE),
    }


def run_prediction(X, y):
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=TEST_SIZE,
                                              stratify=y, random_state=RANDOM_STATE)
    rows, fitted = [], {}
    for name, clf in get_models().items():
        pipe = Pipeline([("pre", build_preprocessor()), ("clf", clf)])
        cv = cross_val_score(pipe, X_tr, y_tr, cv=5, scoring="f1_macro", n_jobs=-1)
        pipe.fit(X_tr, y_tr)
        pred = pipe.predict(X_te)
        p, r, f, _ = precision_recall_fscore_support(y_te, pred, average="macro",
                                                     zero_division=0)
        rows.append({"model": name, "cv_f1_macro(5-fold)": round(cv.mean(), 4),
                     "accuracy": round(accuracy_score(y_te, pred), 4),
                     "precision_macro": round(p, 4), "recall_macro": round(r, 4),
                     "f1_macro": round(f, 4)})
        fitted[name] = pipe

    results = pd.DataFrame(rows).sort_values("f1_macro", ascending=False)
    best_name = results.iloc[0]["model"]
    best = fitted[best_name]
    y_pred = best.predict(X_te)

    # --- comparison plot
    fig, ax = plt.subplots(figsize=(8, 4.5))
    results.set_index("model")[["accuracy", "f1_macro"]].plot(kind="bar", ax=ax)
    ax.set_ylim(0, 1.05); ax.set_title("Model comparison — travel behavior prediction")
    plt.xticks(rotation=15)
    save_plot(fig, "11_model_comparison.png")

    # --- confusion matrix
    labels = sorted(y.unique())
    cm = confusion_matrix(y_te, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels,
                yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(f"Confusion matrix — {best_name}")
    save_plot(fig, "12_confusion_matrix.png")

    # --- feature importance (tree-based models)
    if hasattr(best.named_steps["clf"], "feature_importances_"):
        names = [n.split("__", 1)[-1]
                 for n in best.named_steps["pre"].get_feature_names_out()]
        imp = (pd.Series(best.named_steps["clf"].feature_importances_, index=names)
               .sort_values(ascending=False).head(15))
        fig, ax = plt.subplots(figsize=(7, 5.5))
        sns.barplot(x=imp.values, y=imp.index, ax=ax, color="#4c72b0")
        ax.set_title(f"Top features — {best_name}")
        save_plot(fig, "13_feature_importance.png")

    with open(REPORTS_DIR / "prediction_report.txt", "w") as f:
        f.write(results.to_string(index=False))
        f.write(f"\n\nBest model: {best_name}\n\n")
        f.write(classification_report(y_te, y_pred, zero_division=0))
    print(f"[PREDICTION] best model = {best_name} "
          f"(test f1_macro={results.iloc[0]['f1_macro']:.3f})")
    return {"results": results, "best_name": best_name, "best_model": best,
            "y_test": y_te, "y_pred": y_pred}