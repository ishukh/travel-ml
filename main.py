"""End-to-end pipeline: data → preprocessing → EDA → clustering → prediction
→ recommendation → evaluation → saved bundle for the Streamlit app."""
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.clustering import run_clustering
from src.config import MODELS_DIR, REPORTS_DIR, TOP_N_RECOMMENDATIONS
from src.data_generation import load_or_generate
from src.eda import run_eda
from src.evaluation import recommender_metrics
from src.prediction import run_prediction
from src.preprocessing import clean_travelers, fit_mlb, make_feature_frame
from src.recommender import (TravelRecommender, build_itinerary,
                             interest_weight_vector)


def popularity_baseline_metrics(train_r, test_r, k=10, threshold=4.0):
    """Baseline for comparison: rank destinations by mean training rating."""
    seen = train_r.groupby("traveler_id")["destination_id"].apply(set)
    rel = (test_r[test_r["rating"] >= threshold]
           .groupby("traveler_id")["destination_id"].apply(set))
    pop = (train_r.groupby("destination_id")["rating"].mean()
           .sort_values(ascending=False).index.tolist())
    prec, rec = [], []
    for u, relevant in rel.items():
        top = [d for d in pop if d not in seen.get(u, set())][:k]
        hits = len(set(top) & relevant)
        prec.append(hits / k)
        rec.append(hits / len(relevant))
    return {"precision@10": round(float(np.mean(prec)), 4),
            "recall@10": round(float(np.mean(rec)), 4)}


def demo_recommendation(bundle):
    """Console demo: take one traveler, predict profile, recommend, plan trip."""
    travelers = bundle["travelers"]
    t = travelers.iloc[[100]]
    X = make_feature_frame(t, bundle["mlb"])
    probs = bundle["classifier"].predict_proba(X)[0]
    type_probs = dict(zip(bundle["classifier"].classes_, probs))
    cluster = int(bundle["kmeans"].predict(X)[0])
    profile = {"type_probs": type_probs, "interests": t.iloc[0]["interests"],
               "cluster_id": cluster,
               "budget_per_day": float(t.iloc[0]["budget_per_day"]),
               "season": t.iloc[0]["preferred_season"],
               "with_children": bool(t.iloc[0]["with_children"]),
               "user_ratings": {}, "type_centroids": bundle["type_centroids"]}
    recs = bundle["recommender"].recommend(profile, top_n=3)
    print("\n--- DEMO ---")
    print("Predicted type:", max(type_probs, key=type_probs.get), "| cluster:", cluster)
    print(recs[["name", "country", "avg_daily_cost", "hybrid_score",
                "why_recommended"]].to_string(index=False))
    iw = interest_weight_vector(type_probs, t.iloc[0]["interests"],
                                bundle["type_centroids"])
    days = int(t.iloc[0]["trip_duration_days"])
    for day in build_itinerary(recs.iloc[0], iw, min(days, 5)):
        print(f"  Day {day['day']}: {day['morning']} | {day['afternoon']} | {day['evening']}")


def main():
    print("== 1. Data collection & preparation ==")
    travelers_raw, destinations, ratings = load_or_generate()
    travelers = clean_travelers(travelers_raw)
    print(f"travelers {travelers.shape} | destinations {destinations.shape} "
          f"| ratings {ratings.shape}")

    print("== 2. Exploratory Data Analysis ==")
    run_eda(travelers, destinations, ratings)

    print("== 3. Feature engineering ==")
    mlb = fit_mlb(travelers["interests"])
    X = make_feature_frame(travelers, mlb)
    y = travelers["traveler_type"]
    print(f"feature matrix: {X.shape}")

    print("== 4. Traveler segmentation (K-Means) ==")
    cl = run_clustering(X, travelers)
    travelers["cluster"] = cl["labels"]
    print(cl["metrics"].round(4).to_string(index=False))
    print(f"selected k={cl['best_k']} | "
          f"ARI vs declared traveler type = {cl['ari']:.3f}")

    print("== 5. Travel behavior prediction (supervised) ==")
    pred = run_prediction(X, y)
    print(pred["results"].to_string(index=False))

    print("== 6. Hybrid recommender + evaluation ==")
    train_r, test_r = train_test_split(ratings, test_size=0.15, random_state=42)
    cluster_of = pd.Series(cl["labels"], index=travelers["traveler_id"])
    rec = TravelRecommender(destinations, train_r).fit(cluster_of)
    rmet = recommender_metrics(rec.pred_matrix, test_r, rec.R, k=10)
    base = popularity_baseline_metrics(train_r, test_r, k=10)
    print("Hybrid SVD recommender:", json.dumps(rmet))
    print("Popularity baseline   :", json.dumps(base))

    interest_cols = [c for c in X.columns if c.startswith("interest_")]
    type_centroids = (pd.concat([X[interest_cols], y], axis=1)
                      .groupby("traveler_type").mean())

    bundle = {"mlb": mlb, "kmeans": cl["model"], "classifier": pred["best_model"],
              "model_name": pred["best_name"], "recommender": rec,
              "type_centroids": type_centroids, "cluster_profiles": cl["profiles"],
              "destinations": destinations, "travelers": travelers,
              "training_results": pred["results"],
              "clustering_metrics": cl["metrics"], "rec_metrics": rmet,
              "top_n": TOP_N_RECOMMENDATIONS}
    joblib.dump(bundle, MODELS_DIR / "travel_bundle.joblib")
    with open(REPORTS_DIR / "recommender_metrics.json", "w") as f:
        json.dump({"hybrid_svd": rmet, "popularity_baseline": base}, f, indent=2)
    print(f"\nSaved bundle → {MODELS_DIR / 'travel_bundle.joblib'}")

    demo_recommendation(bundle)


if __name__ == "__main__":
    main()

# END OF main.py — if you can read this line in your editor, the paste is complete.