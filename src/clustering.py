"""Traveler segmentation with K-Means (k selected by silhouette + elbow)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (adjusted_rand_score, calinski_harabasz_score,
                             davies_bouldin_score, silhouette_score)
from sklearn.pipeline import Pipeline

from src.config import RANDOM_STATE, REPORTS_DIR
from src.eda import save_plot
from src.preprocessing import build_preprocessor


def run_clustering(X, df, k_range=range(2, 9)):
    # ---- k selection loop
    metrics = []
    for k in k_range:
        pipe = Pipeline([("pre", build_preprocessor()),
                         ("kmeans", KMeans(n_clusters=k, n_init=10,
                                           random_state=RANDOM_STATE))])
        labels = pipe.fit_predict(X)
        Xs = pipe.named_steps["pre"].transform(X)
        metrics.append({"k": k,
                        "silhouette": silhouette_score(Xs, labels),
                        "calinski_harabasz": calinski_harabasz_score(Xs, labels),
                        "davies_bouldin": davies_bouldin_score(Xs, labels),
                        "inertia": pipe.named_steps["kmeans"].inertia_})
    res = pd.DataFrame(metrics)
    best_k = int(res.loc[res["silhouette"].idxmax(), "k"])

    # ---- final model at best_k
    final = Pipeline([("pre", build_preprocessor()),
                      ("kmeans", KMeans(n_clusters=best_k, n_init=10,
                                        random_state=RANDOM_STATE))]).fit(X)
    labels = final.named_steps["kmeans"].labels_
    Xs = final.named_steps["pre"].transform(X)

    # ---- elbow + silhouette plot
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(res["k"], res["inertia"], "o-")
    axes[0].set_title("Elbow (inertia)")
    axes[0].set_xlabel("k")
    axes[1].plot(res["k"], res["silhouette"], "o-", color="#c44e52")
    axes[1].axvline(best_k, ls="--", c="gray")
    axes[1].set_title("Silhouette score")
    axes[1].set_xlabel("k")
    save_plot(fig, "09_kmeans_k_selection.png")

    # ---- PCA 2-D cluster map
    pts = PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(Xs)
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    sc = ax.scatter(pts[:, 0], pts[:, 1], c=labels, cmap="tab10", s=14, alpha=.75)
    ax.legend(*sc.legend_elements(), title="cluster")
    ax.set_title(f"K-Means segments (k={best_k}, PCA projection)")
    save_plot(fig, "10_pca_clusters.png")

    # ---- segment profiles & validation vs declared traveler type
    d = df.copy()
    d["cluster"] = labels
    prof = d.groupby("cluster").agg(
        size=("traveler_id", "count"), mean_age=("age", "mean"),
        mean_budget=("budget_per_day", "mean"),
        mean_duration=("trip_duration_days", "mean"),
        mean_trips_per_year=("trips_per_year", "mean"),
        pct_with_children=("with_children", "mean"),
        dominant_type=("traveler_type", lambda s: s.mode()[0]))
    for c in ["transport_pref", "accommodation_pref", "preferred_season"]:
        prof[c] = d.groupby("cluster")[c].agg(lambda s: s.mode()[0])
    top_int = (d.explode("interests").groupby(["cluster", "interests"]).size()
               .reset_index(name="n")
               .sort_values(["cluster", "n"], ascending=[True, False])
               .groupby("cluster").head(3)
               .groupby("cluster")["interests"].apply(lambda s: ", ".join(s))
               .rename("top_interests"))
    prof = prof.join(top_int)
    ct = pd.crosstab(d["cluster"], d["traveler_type"], normalize="index").round(3)
    ari = adjusted_rand_score(d["traveler_type"], labels)

    res.to_csv(REPORTS_DIR / "clustering_metrics.csv", index=False)
    prof.round(2).to_csv(REPORTS_DIR / "cluster_profiles.csv")
    ct.to_csv(REPORTS_DIR / "cluster_vs_type_crosstab.csv")

    return {"model": final, "labels": labels, "metrics": res, "best_k": best_k,
            "X_scaled": Xs, "profiles": prof, "type_crosstab": ct, "ari": ari}

# END OF clustering.py — if you can read this line in your editor, the paste is complete.