"""Interactive UI: predict travel profile, recommend destinations, build itinerary.
Run:  streamlit run app.py   (after `python main.py` once)"""
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from src.config import (ACCOMMODATION_OPTIONS, HYBRID_WEIGHTS, INTERESTS,
                        MODELS_DIR, PLOTS_DIR, SEASONS, TRANSPORT_OPTIONS)
from src.preprocessing import make_feature_frame
from src.recommender import build_itinerary, interest_weight_vector

FLAGS = {"Indonesia": "🇮🇩", "Japan": "🇯🇵", "Switzerland": "🇨🇭", "France": "🇫🇷",
         "Italy": "🇮🇹", "Spain": "🇪🇸", "Iceland": "🇮🇸", "New Zealand": "🇳🇿",
         "Morocco": "🇲🇦", "Thailand": "🇹🇭", "Greece": "🇬🇷", "South Africa": "🇿🇦",
         "Tanzania": "🇹🇿", "Peru": "🇵🇪", "UAE": "🇦🇪", "USA": "🇺🇸",
         "Canada": "🇨🇦", "Czech Republic": "🇨🇿", "India": "🇮🇳", "Turkey": "🇹🇷",
         "Maldives": "🇲🇻", "Vietnam": "🇻🇳"}

PRESETS = {
    "🏔️ Adventure trip": dict(age=27, income=4200, budget=140, duration=6,
                              trips_yr=3.5, season="summer", transport="flight",
                              accom="homestay", group=2, children=False,
                              interests=["adventure", "nature", "photography"]),
    "👨‍👩‍👧 Family holiday": dict(age=38, income=6500, budget=200, duration=8,
                                trips_yr=1.5, season="summer", transport="car",
                                accom="resort", group=4, children=True,
                                interests=["nature", "wildlife", "relaxation"]),
    "🎒 Budget backpacking": dict(age=22, income=2200, budget=55, duration=12,
                                  trips_yr=5.0, season="winter", transport="bus",
                                  accom="hostel", group=1, children=False,
                                  interests=["nightlife", "adventure", "food"]),
    "💆 Relaxation getaway": dict(age=45, income=5500, budget=180, duration=9,
                                  trips_yr=2.5, season="spring", transport="flight",
                                  accom="resort", group=2, children=False,
                                  interests=["relaxation", "food", "photography"]),
}


def apply_preset(name):
    for k, v in PRESETS[name].items():
        st.session_state[k] = v


def radar_chart(weights: dict):
    labels = list(INTERESTS)
    vals = [float(weights.get(i, 0.0)) for i in labels]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    fig = plt.figure(figsize=(4.0, 4.0))
    ax = plt.subplot(111, polar=True)
    ax.plot(angles + angles[:1], vals + vals[:1], color="#4c72b0", linewidth=2)
    ax.fill(angles + angles[:1], vals + vals[:1], color="#4c72b0", alpha=0.25)
    ax.set_xticks(angles)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_yticklabels([])
    ax.set_title("Your interest profile (ML-learned weights)", fontsize=10, pad=14)
    return fig


st.set_page_config(page_title="AI Travel Planner", page_icon="🧭", layout="wide")
st.markdown("""<style>.block-container{padding-top:1.4rem;}</style>""",
            unsafe_allow_html=True)


@st.cache_resource
def load_bundle():
    return joblib.load(MODELS_DIR / "travel_bundle.joblib")


try:
    bundle = load_bundle()
except Exception:
    st.error("Model bundle not found. Run `python main.py` once to train, "
             "then refresh this page.")
    st.stop()

rm = bundle.get("rec_metrics", {})
st.markdown(f"""
<div style="background:linear-gradient(135deg,#1e3c72 0%,#2a5298 55%,#4b9fd5 100%);
            border-radius:16px;padding:26px 30px;margin-bottom:16px;">
  <div style="font-size:32px;font-weight:800;color:#ffffff;">🧭 AI Travel Planner</div>
  <div style="color:#e8f1ff;font-size:15px;margin-top:4px;">
    Travel behavior prediction & personalized itineraries — powered by K-Means
    segmentation, {bundle['model_name']} classification, and a hybrid recommender.</div>
  <div style="margin-top:12px;">
    <span style="background:rgba(255,255,255,.18);color:#fff;border-radius:20px;
                 padding:4px 12px;font-size:12px;margin-right:8px;">🤖 {bundle['model_name']}</span>
    <span style="background:rgba(255,255,255,.18);color:#fff;border-radius:20px;
                 padding:4px 12px;font-size:12px;margin-right:8px;">🧩 {len(bundle['cluster_profiles'])} segments</span>
    <span style="background:rgba(255,255,255,.18);color:#fff;border-radius:20px;
                 padding:4px 12px;font-size:12px;margin-right:8px;">🌍 {len(bundle['destinations'])} destinations</span>
    <span style="background:rgba(255,255,255,.18);color:#fff;border-radius:20px;
                 padding:4px 12px;font-size:12px;">🎯 RMSE {rm.get('rmse', '—')}</span>
  </div>
</div>""", unsafe_allow_html=True)

# ------------------------------------------------------------ sidebar inputs
st.sidebar.header("⚡ Quick presets")
for name in PRESETS:
    st.sidebar.button(name, on_click=apply_preset, args=(name,),
                      use_container_width=True)

st.sidebar.divider()
st.sidebar.header("🧳 Your travel preferences")
age = st.sidebar.slider("Age", 18, 75, 30, key="age")
income = st.sidebar.slider("Monthly income (USD)", 500, 15000, 3500, 100, key="income")
budget = st.sidebar.slider("Daily budget (USD)", 30, 500, 150, key="budget")
duration = st.sidebar.slider("Trip duration (days)", 2, 21, 7, key="duration")
trips_yr = st.sidebar.slider("Trips per year (past behavior)", 0.5, 12.0, 3.0, 0.5,
                             key="trips_yr")
season = st.sidebar.selectbox("Preferred season", SEASONS, key="season")
transport = st.sidebar.selectbox("Transport preference", TRANSPORT_OPTIONS,
                                 key="transport")
accom = st.sidebar.selectbox("Accommodation preference", ACCOMMODATION_OPTIONS,
                             key="accom")
group = st.sidebar.slider("Group size", 1, 8, 2, key="group")
children = st.sidebar.checkbox("Traveling with children", key="children")
interests = st.sidebar.multiselect("Interests (pick 1+)", INTERESTS,
                                   default=["culture", "food"], key="interests")
st.sidebar.caption(f"💰 Estimated total spend ≈ **${budget * duration:,.0f}** "
                   f"({duration} days × ${budget})")

st.sidebar.divider()
st.sidebar.subheader("⭐ Rate places you've visited")
if "ratings" not in st.session_state:
    st.session_state.ratings = {}
name_map = dict(zip(bundle["destinations"]["destination_id"],
                    bundle["destinations"]["name"]))
pick = st.sidebar.selectbox("Destination",
                            bundle["destinations"]["name"].tolist(), key="rate_pick")
val = st.sidebar.slider("Your rating", 1, 5, 4, key="rate_val")
if st.sidebar.button("Add rating", use_container_width=True):
    d_id = bundle["destinations"].loc[
        bundle["destinations"]["name"] == pick, "destination_id"].iloc[0]
    st.session_state.ratings[d_id] = val
if st.session_state.ratings:
    st.sidebar.caption("Your ratings: " + ", ".join(
        f"{name_map[k]} {v}★" for k, v in st.session_state.ratings.items()))
    if st.sidebar.button("Clear all ratings", use_container_width=True):
        st.session_state.ratings = {}

# ------------------------------------------- live inference (updates instantly)
user = pd.DataFrame([{
    "age": age, "gender": "Other", "occupation": "employed",
    "monthly_income": float(income), "budget_per_day": float(budget),
    "trip_duration_days": duration, "trips_per_year": float(trips_yr),
    "group_size": group, "with_children": children,
    "interests": interests or ["culture"], "preferred_season": season,
    "transport_pref": transport, "accommodation_pref": accom,
    "traveler_id": "NEW"}])
X_user = make_feature_frame(user, bundle["mlb"])

probs = bundle["classifier"].predict_proba(X_user)[0]
type_probs = dict(zip(bundle["classifier"].classes_, probs))
cluster = int(bundle["kmeans"].predict(X_user)[0])
pred_type = max(type_probs, key=type_probs.get)
iw = interest_weight_vector(type_probs, interests or ["culture"],
                            bundle["type_centroids"])

tab_plan, tab_explore, tab_insights, tab_about = st.tabs(
    ["🎯 Plan my trip", "🌍 Explore destinations", "📈 Model insights", "ℹ️ How it works"])

# ------------------------------------------------------------------ PLAN TAB
with tab_plan:
    c1, c2, c3 = st.columns([3, 3, 2])
    with c1:
        st.subheader("🔮 Predicted profile")
        st.markdown(f"## {pred_type}")
        st.caption(f"Classified live by {bundle['model_name']}")
        for tname, p in sorted(type_probs.items(), key=lambda kv: -kv[1]):
            st.write(f"**{tname}** — {p * 100:.0f}%")
            st.progress(float(min(p, 1.0)))
    with c2:
        st.subheader(f"🧩 Your segment: cluster {cluster}")
        prof = bundle["cluster_profiles"]
        if cluster in prof.index:
            p = prof.loc[cluster]
            st.markdown(
                f"**Dominant type:** {p['dominant_type']}  \n"
                f"**Size:** {int(p['size'])} similar travelers  \n"
                f"**Avg budget/day:** ${p['mean_budget']:.0f} · "
                f"**Avg trip:** {p['mean_duration']:.1f} days · "
                f"**Trips/yr:** {p['mean_trips_per_year']:.1f}  \n"
                f"**Typical style:** {p['transport_pref']} + {p['accommodation_pref']}  \n"
                f"**Top interests:** {p['top_interests']}")
    with c3:
        st.pyplot(radar_chart(iw), use_container_width=True)

    st.divider()
    profile = {"type_probs": type_probs, "interests": interests or ["culture"],
               "cluster_id": cluster, "budget_per_day": float(budget),
               "season": season, "with_children": children,
               "user_ratings": st.session_state.ratings,
               "type_centroids": bundle["type_centroids"]}
    recs = bundle["recommender"].recommend(profile, top_n=bundle.get("top_n", 5))
    max_h = float(recs["hybrid_score"].max()) or 1.0

    st.subheader("🎯 Recommended for you")
    medals = ["🥇", "🥈", "🥉"]
    for i, (_, row) in enumerate(recs.iterrows()):
        with st.container(border=True):
            h1, h2 = st.columns([4, 1.4])
            with h1:
                flag = FLAGS.get(row["country"], "📍")
                st.markdown(f"**{medals[i] if i < 3 else f'{i + 1}.'} "
                            f"{flag} {row['name']}** · {row['country']} "
                            f"({row['region']})")
                st.caption(" ".join(f"`{t}`" for t in row["tags"]) + "  |  Best in: "
                           + ", ".join(row["best_seasons"]))
                st.write("💡 " + row["why_recommended"])
            with h2:
                st.metric("Match score", f"{row['hybrid_score']:.2f}")
                st.progress(float(min(row["hybrid_score"] / max_h, 1.0)))
                st.caption(f"≈ ${row['avg_daily_cost']:.0f}/day · "
                           f"{row['budget_tier']} tier"
                           + (" · 👨‍👩‍👧 family-friendly" if row["family_friendly"] else ""))
        with st.expander(f"Score breakdown — {row['name']}"):
            st.bar_chart(pd.DataFrame({
                "content": [row["content_score"]],
                "segment": [row["segment_score"]],
                "hybrid": [row["hybrid_score"]]}, index=[row["name"]]))

    st.divider()
    st.subheader("🗺️ Your day-by-day itinerary")
    choice = st.selectbox("Build itinerary for:", recs["name"].tolist())
    sel = recs[recs["name"] == choice].iloc[0]
    total_cost = float(sel["avg_daily_cost"]) * duration
    st.caption(f"✈️ {duration} days in {choice} · estimated on-ground cost ≈ "
               f"**${total_cost:,.0f}** (${sel['avg_daily_cost']:.0f}/day, excl. flights)")
    plan = pd.DataFrame(build_itinerary(sel, iw, duration))
    for _, day in plan.iterrows():
        with st.container(border=True):
            st.markdown(f"**📅 Day {day['day']}**")
            m, a, e = st.columns(3)
            m.markdown(f"🌅 **Morning**  \n{day['morning']}")
            a.markdown(f"☀️ **Afternoon**  \n{day['afternoon']}")
            e.markdown(f"🌙 **Evening**  \n{day['evening']}")
    st.download_button("⬇️ Download itinerary (CSV)",
                       plan.to_csv(index=False).encode("utf-8"),
                       file_name=f"itinerary_{choice.replace(' ', '_').lower()}.csv",
                       mime="text/csv")

# -------------------------------------------------------------- EXPLORE TAB
with tab_explore:
    st.subheader("🌍 Destination catalog")
    st.caption("Browse all destinations — including an ML-driven ranking based on "
               "how travelers in *your* segment actually rated each place.")
    dest = bundle["destinations"].copy()
    dest["avg_rating"] = dest["destination_id"].map(
        bundle["recommender"].R.mean(axis=0)).round(2)
    dest["segment_rating"] = dest["destination_id"].map(
        bundle["recommender"].cluster_affinity.loc[cluster]).round(2)

    f1, f2, f3, f4 = st.columns(4)
    regions = f1.multiselect("Region", sorted(dest["region"].unique()))
    tiers = f2.multiselect("Budget tier", ["low", "medium", "high"])
    f_int = f3.selectbox("Interest", ["Any"] + INTERESTS)
    fam_only = f4.checkbox("Family-friendly only")

    view = dest
    if regions:
        view = view[view["region"].isin(regions)]
    if tiers:
        view = view[view["budget_tier"].isin(tiers)]
    if f_int != "Any":
        view = view[view["interests"].apply(lambda L: f_int in L)]
    if fam_only:
        view = view[view["family_friendly"]]

    sort = st.radio("Sort by", ["Avg rating (all travelers)", "Best for your segment",
                                "Popularity", "Cheapest per day"],
                    horizontal=True)
    if sort == "Best for your segment":
        view = view.sort_values("segment_rating", ascending=False)
    elif sort == "Popularity":
        view = view.sort_values("popularity", ascending=False)
    elif sort == "Cheapest per day":
        view = view.sort_values("avg_daily_cost")
    else:
        view = view.sort_values("avg_rating", ascending=False)

    st.caption(f"{len(view)} destinations shown")
    try:
        st.dataframe(
            view[["name", "country", "region", "tags", "interests", "avg_daily_cost",
                  "avg_rating", "segment_rating", "popularity", "best_seasons",
                  "family_friendly"]],
            use_container_width=True, hide_index=True,
            column_config={
                "avg_daily_cost": st.column_config.NumberColumn("Cost/day", format="$%.0f"),
                "avg_rating": st.column_config.ProgressColumn("Avg rating ★",
                                                              min_value=1.0, max_value=5.0,
                                                              format="%.2f"),
                "segment_rating": st.column_config.ProgressColumn("Your segment ★",
                                                                  min_value=1.0, max_value=5.0,
                                                                  format="%.2f"),
                "popularity": st.column_config.ProgressColumn("Popularity",
                                                              min_value=0.0, max_value=1.0,
                                                              format="%.2f")})
    except Exception:
        st.dataframe(view, use_container_width=True)

# ------------------------------------------------------------- INSIGHTS TAB
with tab_insights:
    acc = bundle["training_results"].iloc[0]
    k_metrics = bundle["clustering_metrics"]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Best classifier F1 (macro)", f"{acc['f1_macro']:.3f}", bundle["model_name"])
    m2.metric("Accuracy", f"{acc['accuracy']:.3f}")
    m3.metric("Best silhouette (k)",
              f"{k_metrics['silhouette'].max():.3f}", f"k={k_metrics.loc[k_metrics['silhouette'].idxmax(), 'k']}")
    m4.metric("Recommender RMSE", rm.get("rmse", "—"))

    st.subheader("1️⃣ Supervised model comparison (traveler-type prediction)")
    st.dataframe(bundle["training_results"], use_container_width=True)
    st.success(f"Model deployed in this app: **{bundle['model_name']}**")

    st.subheader("2️⃣ Segmentation profiles (K-Means)")
    st.dataframe(bundle["cluster_profiles"].round(2), use_container_width=True)

    st.subheader("3️⃣ Clustering & recommender metrics")
    t1, t2 = st.columns(2)
    t1.caption("K selection")
    t1.dataframe(k_metrics.round(4), use_container_width=True)
    t2.caption("Hybrid recommender")
    t2.dataframe(pd.DataFrame([rm]), use_container_width=True)

    plots = sorted(PLOTS_DIR.glob("*.png"))
    if plots:
        st.subheader("4️⃣ Training-time visualizations")
        for p in plots:
            with st.expander(p.stem.replace("_", " ")):
                st.image(str(p))

# ---------------------------------------------------------------- ABOUT TAB
with tab_about:
    w = HYBRID_WEIGHTS
    st.markdown(
        "This app is the **interactive layer** of an ML pipeline trained offline by `main.py`:\n\n"
        "```\n"
        "Travel dataset → preprocessing → EDA → feature engineering → K-Means segmentation\n"
        f"→ {bundle['model_name']} behavior classifier → hybrid recommender → itinerary builder\n"
        "```\n\n"
        "- **Segmentation:** K-Means clustering on demographics, budget, duration, "
        "preferences and interests\n"
        f"- **Prediction:** {bundle['model_name']} classifies your traveler type (5 classes)\n"
        "- **Recommendation scoring:**\n"
        f"  - {w['content']:.0%} **content-based** — cosine match between your interest "
        "profile and each destination's interest vector\n"
        f"  - {w['collaborative']:.0%} **collaborative filtering** — TruncatedSVD matrix "
        "factorization over the traveler×destination rating matrix (your ratings are "
        "folded in live)\n"
        f"  - {w['segment']:.0%} **segment affinity** — how travelers in *your cluster* "
        "actually rated each destination\n"
        "- **Cold start:** with no ratings given, scoring falls back to content + segment signals\n"
        "- **Constraints:** budget overruns and non-family-friendly destinations are "
        "penalized; your chosen season gets a boost\n\n"
        "**Try this demo flow:** click a preset → watch profile + recommendations update "
        "→ rate 2–3 places in the sidebar → see collaborative filtering reshuffle the "
        "ranking → download your itinerary CSV."
    )

st.caption("🧭 AI Travel Planner — ML coursework project · "
           "pipeline: dataset → preprocessing → EDA → K-Means → classification → hybrid recommender")

# END OF app.py — if you can read this line in your editor, the paste is complete.