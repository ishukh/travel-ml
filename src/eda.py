"""Exploratory Data Analysis — saves all plots and a summary report."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.config import PLOTS_DIR, REPORTS_DIR
from src.preprocessing import NUMERIC_FEATURES


def save_plot(fig, name):
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / name, dpi=140, bbox_inches="tight")
    plt.close(fig)


def run_eda(travelers, destinations, ratings):
    sns.set_theme(style="whitegrid")

    # 1. Target distribution
    fig, ax = plt.subplots(figsize=(8, 4))
    travelers["traveler_type"].value_counts().plot(kind="bar", ax=ax, color="#4c72b0")
    ax.set_title("Traveler type distribution")
    ax.set_xlabel("")
    save_plot(fig, "01_target_distribution.png")

    # 2. Age & budget by traveler type
       # 2. Age & budget by traveler type
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    sns.boxplot(data=travelers, x="traveler_type", y="age",
                hue="traveler_type", legend=False, palette="Set2", ax=axes[0])
    axes[0].tick_params(axis="x", rotation=20)
    axes[0].set_title("Age by traveler type")
    sns.boxplot(data=travelers, x="traveler_type", y="budget_per_day",
                hue="traveler_type", legend=False, palette="Set2", ax=axes[1])
    axes[1].tick_params(axis="x", rotation=20)
    axes[1].set_title("Daily budget (USD) by traveler type")
    save_plot(fig, "02_age_budget_by_type.png")

    # 3. Interest frequency + interest profile per type
    exploded = travelers[["traveler_type", "interests"]].explode("interests")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    exploded["interests"].value_counts().plot(kind="barh", ax=axes[0], color="#55a868")
    axes[0].set_title("Interest frequency (all travelers)")
    heat = pd.crosstab(exploded["traveler_type"], exploded["interests"],
                       normalize="index")
    sns.heatmap(heat, annot=True, fmt=".2f", cmap="YlGnBu", ax=axes[1])
    axes[1].set_title("Mean interest intensity per traveler type")
    save_plot(fig, "03_interests.png")

    # 4. Correlation heatmap (numeric)
    fig, ax = plt.subplots(figsize=(7, 5.5))
    sns.heatmap(travelers[NUMERIC_FEATURES + ["with_children"]].corr(),
                annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Numeric feature correlations")
    save_plot(fig, "04_numeric_correlations.png")

    # 5. Season preference per type (stacked)
    ct = pd.crosstab(travelers["traveler_type"], travelers["preferred_season"],
                     normalize="index")
    ct.plot(kind="bar", stacked=True, figsize=(8, 4.5), colormap="viridis")
    plt.title("Preferred season by traveler type")
    plt.xlabel("")
    plt.xticks(rotation=20, ha="right")
    save_plot(fig, "05_season_by_type.png")

    # 6. Budget vs duration coloured by type
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(data=travelers, x="budget_per_day", y="trip_duration_days",
                    hue="traveler_type", alpha=.6, s=25, ax=ax)
    ax.set_title("Budget vs trip duration")
    save_plot(fig, "06_budget_vs_duration.png")

    # 7. Destination popularity (mean observed rating)
    avg = (ratings.groupby("destination_id")["rating"].mean()
           .rename("avg_rating").reset_index()
           .merge(destinations[["destination_id", "name"]], on="destination_id")
           .sort_values("avg_rating").tail(12))
    fig, ax = plt.subplots(figsize=(8, 5))
    plt.barh(avg["name"], avg["avg_rating"], color="#c44e52")
    plt.title("Top-12 destinations by average rating")
    plt.xlim(1, 5)
    save_plot(fig, "07_top_destinations.png")

    # 8. Rating distribution
    fig, ax = plt.subplots(figsize=(6, 3.5))
    sns.countplot(x="rating", data=ratings, ax=ax, color="#8172b2")
    ax.set_title("Rating distribution")
    save_plot(fig, "08_rating_distribution.png")

    with open(REPORTS_DIR / "eda_summary.txt", "w") as f:
        f.write("=== Travelers describe ===\n"
                + travelers.describe(include="all").to_string())
        f.write("\n\n=== Traveler type counts ===\n"
                + travelers["traveler_type"].value_counts().to_string())
        f.write("\n\n=== Ratings describe ===\n" + ratings.describe().to_string())
    print(f"[EDA] {len(list(PLOTS_DIR.glob('*.png')))} plots → {PLOTS_DIR}")