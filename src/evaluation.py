"""Model evaluation helpers (classification metrics live in prediction.py)."""
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


def recommender_metrics(pred_matrix, test_ratings, train_matrix, k=10, threshold=4.0):
    """RMSE/MAE on held-out ratings + Precision@k / Recall@k ranking metrics
    (relevant = held-out rating >= threshold, already-rated items excluded)."""
    truth = dict(zip(zip(test_ratings["traveler_id"], test_ratings["destination_id"]),
                     test_ratings["rating"]))
    preds, actuals = [], []
    for (u, i), r in truth.items():
        if u in pred_matrix.index and i in pred_matrix.columns:
            preds.append(pred_matrix.at[u, i]); actuals.append(r)
    rmse = float(np.sqrt(mean_squared_error(actuals, preds)))
    mae = float(mean_absolute_error(actuals, preds))

    precisions, recalls = [], []
    for u in test_ratings["traveler_id"].unique():
        if u not in pred_matrix.index:
            continue
        seen = set(train_matrix.loc[u].dropna().index)
        relevant = (set(test_ratings[(test_ratings["traveler_id"] == u) &
                                     (test_ratings["rating"] >= threshold)]["destination_id"])
                    - seen)
        if not relevant:
            continue
        s = pred_matrix.loc[u].copy()
        s.loc[list(seen)] = np.nan
        top = s.dropna().sort_values(ascending=False).head(k).index
        hits = len(set(top) & relevant)
        precisions.append(hits / k)
        recalls.append(hits / len(relevant))
    return {"rmse": round(rmse, 4), "mae": round(mae, 4),
            f"precision@{k}": round(float(np.mean(precisions)), 4) if precisions else None,
            f"recall@{k}": round(float(np.mean(recalls)), 4) if recalls else None}