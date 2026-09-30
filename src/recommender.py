"""
Hybrid personalized recommendation — fully data-driven:

  score(dest) = w_c · content(dest)      cosine similarity between the traveler's
                                         expected interest vector (learned type
                                         centroids + explicit selections) and the
                                         destination's interest vector
              + w_cf · collaborative(dest) TruncatedSVD matrix factorization over
                                         the rating matrix (fold-in for new users)
              + w_s  · segment(dest)     mean rating the destination receives from
                                         travelers in the predicted K-Means segment

Hard/soft constraints (budget, season, family) are applied on top as filters and
score penalties. An itinerary builder maps the traveler's interest weights onto
the destination's activity catalogue, day by day.
"""
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD

from src.config import HYBRID_WEIGHTS, LATENT_DIMS, RANDOM_STATE
from src.data_generation import ACTIVITY_POOLS

ACTIVITY_INTEREST_MAP = {a: it for it, acts in ACTIVITY_POOLS.items() for a in acts}


def interest_weight_vector(type_probs, selected_interests, type_centroids):
    """Expected interest weights = P(type) · learned interest centroid per type,
    boosted by the user's explicit selections. Returns dict interest → weight."""
    w = {}
    for t, p in type_probs.items():
        if t in type_centroids.index:
            for c in type_centroids.columns:
                name = c.replace("interest_", "")
                w[name] = w.get(name, 0.0) + p * float(type_centroids.loc[t, c])
    for s in selected_interests:
        w[s] = w.get(s, 0.0) + 1.0
    tot = sum(w.values()) or 1.0
    return {k: v / tot for k, v in w.items()}


class TravelRecommender:
    def __init__(self, destinations, ratings, latent_dims=LATENT_DIMS):
        self.destinations = destinations.reset_index(drop=True)
        self.ratings = ratings
        self.latent_dims = latent_dims

    # ------------------------------------------------------------------ fit
    def fit(self, cluster_of_traveler: pd.Series):
        dest_ids = self.destinations["destination_id"]
        R = self.ratings.pivot_table(index="traveler_id", columns="destination_id",
                                     values="rating", aggfunc="mean").reindex(columns=dest_ids)
        self.R = R
        user_mean = R.mean(axis=1)
        Rc = R.sub(user_mean, axis=0).fillna(0.0)
        k = max(2, min(self.latent_dims, min(Rc.shape) - 1))
        self.svd = TruncatedSVD(n_components=k, random_state=RANDOM_STATE)
        latent = self.svd.fit_transform(Rc.values)
        self.pred_matrix = (pd.DataFrame(self.svd.inverse_transform(latent),
                                         index=R.index, columns=R.columns)
                            .add(user_mean, axis=0).clip(1, 5))

        # destination interest vectors (L2-normalized) for content-based scoring
        all_int = sorted({i for lst in self.destinations["interests"] for i in lst})
        self.interest_index = {c: j for j, c in enumerate(all_int)}
        M = np.zeros((len(self.destinations), len(all_int)))
        for r_i, lst in enumerate(self.destinations["interests"]):
            for it in lst:
                M[r_i, self.interest_index[it]] = 1.0
        norms = np.linalg.norm(M, axis=1, keepdims=True)
        norms[norms == 0] = 1
        self.dest_vectors = M / norms

        # segment (cluster) × destination affinity from actual ratings
        r = self.ratings.copy()
        r["cluster"] = r["traveler_id"].map(cluster_of_traveler)
        ca = r.pivot_table(index="cluster", columns="destination_id",
                           values="rating", aggfunc="mean").reindex(columns=dest_ids)
        self.cluster_affinity = ca.fillna(self.ratings["rating"].mean())
        self.cluster_affinity_norm = (ca - ca.values.min()) / \
            (ca.values.max() - ca.values.min() + 1e-9)
        self.cluster_affinity_norm = self.cluster_affinity_norm.fillna(0.5)
        return self

    # -------------------------------------------------------------- scorers
    def content_scores_from_profile(self, type_probs, selected_interests,
                                    type_centroids):
        w = interest_weight_vector(type_probs, selected_interests, type_centroids)
        vec = np.zeros(len(self.interest_index))
        for name, val in w.items():
            if name in self.interest_index:
                vec[self.interest_index[name]] = val
        n = np.linalg.norm(vec)
        if n == 0:
            return pd.Series(0.0, index=self.R.columns)
        return pd.Series(self.dest_vectors @ (vec / n), index=self.R.columns)

    def cf_scores_for_new_user(self, user_ratings: dict):
        """Fold-in a new user's ratings into the fitted SVD model (1–5 scale)."""
        user_ratings = {k: v for k, v in (user_ratings or {}).items()
                        if k in self.R.columns and v > 0}
        if not user_ratings or self.svd is None:
            return None
        r = pd.Series(0.0, index=self.R.columns)
        for d_id, val in user_ratings.items():
            r[d_id] = float(val)
        rated = r[r > 0]
        um = rated.mean()
        rc = r.copy()
        rc[rated.index] = rated - um
        latent = rc.values @ self.svd.components_.T
        pred = latent @ self.svd.components_ + um
        return pd.Series(((pred - 1.0) / 4.0).clip(0, 1), index=self.R.columns)

    def segment_scores(self, cluster_id):
        if cluster_id in self.cluster_affinity_norm.index:
            return self.cluster_affinity_norm.loc[cluster_id]
        return pd.Series(0.5, index=self.R.columns)

    # ---------------------------------------------------------- recommend
    def recommend(self, profile: dict, top_n: int = 5) -> pd.DataFrame:
        content = self.content_scores_from_profile(profile["type_probs"],
                                                   profile["interests"],
                                                   profile["type_centroids"])
        seg = self.segment_scores(int(profile["cluster_id"]))
        cf = self.cf_scores_for_new_user(profile.get("user_ratings"))
        if cf is None:                                   # cold-start: content + segment
            total = 0.60 * content + 0.40 * seg
        else:
            total = (HYBRID_WEIGHTS["content"] * content
                     + HYBRID_WEIGHTS["collaborative"] * cf
                     + HYBRID_WEIGHTS["segment"] * seg)
            for d_id in profile.get("user_ratings", {}):  # already-known places
                if d_id in total.index:
                    total[d_id] = -1.0

        d = self.destinations.copy()
        d["content_score"] = d["destination_id"].map(content).round(3)
        d["segment_score"] = d["destination_id"].map(seg).round(3)
        d["hybrid_score"] = d["destination_id"].map(total).round(3)

        # ---- constraints: soft penalties keep results explainable
        budget = float(profile["budget_per_day"])
        season = profile.get("season")
        d.loc[d["avg_daily_cost"] > budget * 1.25, "hybrid_score"] *= 0.5
        if season:
            d.loc[d["best_seasons"].apply(lambda s: season in s), "hybrid_score"] *= 1.10
        if profile.get("with_children"):
            d.loc[~d["family_friendly"], "hybrid_score"] *= 0.25

        # ---- human-readable reasons
        def _reasons(row):
            match = [i for i in profile["interests"] if i in row["interests"]]
            out = []
            if match:
                out.append("matches interests: " + ", ".join(match))
            if season and season in row["best_seasons"]:
                out.append(f"great in {season}")
            out.append(f"~${row['avg_daily_cost']:.0f}/day "
                       + ("(within budget)" if row["avg_daily_cost"] <= budget
                          else "(above budget, penalised)"))
            out.append(f"segment avg rating "
                       f"{self.cluster_affinity.loc[int(profile['cluster_id']), row['destination_id']]:.1f}/5")
            return "; ".join(out)
        d["why_recommended"] = d.apply(_reasons, axis=1)

        return (d.sort_values("hybrid_score", ascending=False).head(top_n)
                 [["destination_id", "name", "country", "region", "tags", "interests",
                   "avg_daily_cost", "budget_tier", "best_seasons", "family_friendly",
                   "activities", "popularity",
                   "hybrid_score", "content_score", "segment_score", "why_recommended"]])


def build_itinerary(dest_row, interest_weights: dict, days: int):
    """Greedy itinerary: rank the destination's activities by the traveler's
    learned interest weights (+ popularity), fill 2 slots/day + evening."""
    activities = list(dest_row.get("activities") or [])
    if not activities:                       # fallback if column absent
        for it in (dest_row.get("interests") or []):
            activities.extend(ACTIVITY_POOLS.get(it, [])[:2])
    popularity = float(dest_row.get("popularity", 0.5) or 0.5)

    scored = []
    for act in activities:
        it = ACTIVITY_INTEREST_MAP.get(act, "culture")
        scored.append((act, interest_weights.get(it, 0.05) + 0.15 * popularity))
    scored.sort(key=lambda x: (-x[1], x[0]))
    chosen = [a for a, _ in scored[:days * 2]]
    plan = []
    for day in range(days):
        m = chosen[2 * day] if 2 * day < len(chosen) else "Free morning — explore at your own pace"
        a = chosen[2 * day + 1] if 2 * day + 1 < len(chosen) else "Afternoon at leisure"
        eve = "Dinner featuring local cuisine"
        if any(ACTIVITY_INTEREST_MAP.get(x) == "nightlife" for x in chosen[2 * day:2 * day + 2]):
            eve = "Night out — bars / night market"
        plan.append({"day": day + 1, "morning": m, "afternoon": a, "evening": eve})
    return plan

# END OF recommender.py — if you can read this line in your editor, the paste is complete.