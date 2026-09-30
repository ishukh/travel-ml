"""
Synthetic travel-survey dataset.

Generates (1) a traveler survey (demographics, stated preferences, past behaviour),
(2) a destination catalogue, (3) a traveler→destination rating matrix.
Ratings are produced by a latent compatibility model (interest overlap, tag
affinity, budget fit, season fit, family fit) + noise, so that all downstream
ML tasks (clustering, classification, collaborative filtering) have genuine
learnable structure.

To use REAL data instead: replace `load_or_generate()` with a loader that
returns (travelers_df, destinations_df, ratings_df) with the same columns.
"""
import numpy as np
import pandas as pd

from src.config import (DATA_DIR, RANDOM_STATE, N_TRAVELERS, INTERESTS,
                        TRAVELER_TYPES)

# ---------------------------------------------------------------- activities
ACTIVITY_POOLS = {
    "adventure":   ["hiking & trekking", "scuba diving", "paragliding",
                    "white-water rafting", "zip-lining", "skiing / snowboarding",
                    "surfing lessons", "rock climbing"],
    "culture":     ["museum visit", "old town walking tour", "local cooking class",
                    "cultural performance", "temple / mosque visit"],
    "food":        ["street food tour", "cooking class", "wine / tea tasting",
                    "local market visit", "fine dining experience"],
    "nature":      ["national park excursion", "waterfall hike", "hot springs visit",
                    "scenic boat trip", "botanical garden walk"],
    "nightlife":   ["night market", "rooftop bars & clubs", "live music venue",
                    "beach party"],
    "relaxation":  ["spa & wellness session", "beach lounging", "yoga class",
                    "resort pool day", "sunset cruise"],
    "shopping":    ["local bazaar shopping", "mall & boutique tour",
                    "artisan craft workshops"],
    "history":     ["ancient ruins tour", "castle / palace visit",
                    "guided heritage walk", "archaeological site visit"],
    "wildlife":    ["safari drive", "bird watching tour", "marine life snorkel tour",
                    "wildlife sanctuary visit"],
    "photography": ["photo walk", "viewpoint hike", "sunset photography session"],
}

# ------------------------------------------------------- latent type profiles
# Each profile drives generation: interest distribution over INTERESTS must sum to 1.
TYPE_PROFILES = {
    "Adventure Seeker": {
        "age": (28, 6), "income": (4200, 1300), "budget": (140, 35),
        "duration": (6, 2), "n_interests": (4, 1), "trips_per_year": (3.5, 1.2),
        "group_size": (1.8, 0.7), "children_prob": 0.12,
        "interests": {"adventure": .35, "nature": .20, "photography": .12,
                      "nightlife": .10, "wildlife": .08, "food": .08,
                      "culture": .04, "relaxation": .02, "shopping": .01, "history": .00},
        "transport": {"flight": .50, "car": .30, "train": .15, "bus": .05},
        "accommodation": {"hotel": .30, "hostel": .20, "homestay": .25,
                          "apartment": .15, "resort": .10},
        "season": {"summer": .35, "winter": .30, "spring": .20, "autumn": .15},
        "tag_affinity": {"adventure": 1.0, "mountain": .9, "nature": .9, "wildlife": .8,
                         "island": .7, "beach": .5, "coastal": .5, "desert": .5,
                         "rural": .5, "city": .35, "cultural": .35, "luxury": .2},
    },
    "Cultural Explorer": {
        "age": (38, 10), "income": (4800, 1500), "budget": (150, 35),
        "duration": (8, 3), "n_interests": (4, 1), "trips_per_year": (3.0, 1.2),
        "group_size": (1.9, 0.7), "children_prob": 0.15,
        "interests": {"culture": .25, "history": .22, "food": .18, "photography": .12,
                      "nature": .08, "shopping": .06, "adventure": .04,
                      "nightlife": .03, "relaxation": .02, "wildlife": .00},
        "transport": {"train": .35, "flight": .35, "car": .25, "bus": .05},
        "accommodation": {"hotel": .40, "homestay": .25, "apartment": .20,
                          "resort": .10, "hostel": .05},
        "season": {"spring": .30, "autumn": .35, "summer": .20, "winter": .15},
        "tag_affinity": {"cultural": 1.0, "city": .9, "rural": .6, "desert": .5,
                         "nature": .5, "mountain": .4, "beach": .3, "island": .3,
                         "coastal": .3, "wildlife": .3, "adventure": .3, "luxury": .3},
    },
    "Leisure & Relaxation": {
        "age": (44, 10), "income": (5500, 1600), "budget": (180, 45),
        "duration": (9, 3), "n_interests": (4, 1), "trips_per_year": (2.5, 1.0),
        "group_size": (2.0, 0.7), "children_prob": 0.20,
        "interests": {"relaxation": .30, "food": .15, "nature": .15, "photography": .10,
                      "shopping": .08, "culture": .08, "nightlife": .05,
                      "adventure": .04, "history": .04, "wildlife": .01},
        "transport": {"flight": .55, "car": .30, "train": .10, "bus": .05},
        "accommodation": {"resort": .45, "hotel": .35, "apartment": .15,
                          "homestay": .05, "hostel": .00},
        "season": {"spring": .30, "autumn": .30, "summer": .25, "winter": .15},
        "tag_affinity": {"island": .9, "beach": .9, "coastal": .9, "luxury": .8,
                         "nature": .6, "city": .5, "mountain": .5, "cultural": .4,
                         "rural": .4, "adventure": .3, "desert": .3, "wildlife": .3},
    },
    "Family Vacationer": {
        "age": (38, 7), "income": (6000, 1800), "budget": (200, 50),
        "duration": (7, 2), "n_interests": (4, 1), "trips_per_year": (1.8, 0.8),
        "group_size": (3.8, 0.8), "children_prob": 0.85,
        "interests": {"nature": .22, "relaxation": .18, "wildlife": .15,
                      "adventure": .12, "culture": .10, "food": .10, "history": .06,
                      "photography": .04, "shopping": .02, "nightlife": .01},
        "transport": {"car": .45, "flight": .45, "train": .08, "bus": .02},
        "accommodation": {"resort": .35, "apartment": .30, "hotel": .30,
                          "homestay": .05, "hostel": .00},
        "season": {"summer": .50, "spring": .20, "autumn": .20, "winter": .10},
        "tag_affinity": {"beach": .9, "nature": .8, "wildlife": .8, "island": .8,
                         "coastal": .7, "city": .6, "mountain": .6, "cultural": .5,
                         "rural": .5, "luxury": .5, "adventure": .4, "desert": .3},
    },
    "Budget Backpacker": {
        "age": (23, 3.5), "income": (2200, 700), "budget": (55, 15),
        "duration": (12, 5), "n_interests": (4, 1), "trips_per_year": (5.0, 2.0),
        "group_size": (1.7, 0.7), "children_prob": 0.05,
        "interests": {"nightlife": .22, "adventure": .20, "food": .18, "culture": .10,
                      "nature": .10, "photography": .08, "relaxation": .05,
                      "shopping": .04, "history": .02, "wildlife": .01},
        "transport": {"bus": .35, "train": .35, "flight": .20, "car": .10},
        "accommodation": {"hostel": .55, "homestay": .20, "apartment": .10,
                          "hotel": .10, "resort": .05},
        "season": {"summer": .35, "winter": .25, "spring": .20, "autumn": .20},
        "tag_affinity": {"city": .8, "adventure": .7, "beach": .7, "cultural": .6,
                         "nature": .6, "island": .6, "rural": .6, "coastal": .6,
                         "mountain": .5, "desert": .4, "wildlife": .4, "luxury": .05},
    },
}

# ------------------------------------------------------ destination catalogue
# (name, country, region, tags, interests, tier, $/day, best seasons, family, popularity)
_DEST = [
    ("Bali","Indonesia","Asia",["island","beach","nature"],["relaxation","adventure","nightlife","food","photography"],"medium",110,["spring","summer","autumn"],True,.90),
    ("Kyoto","Japan","Asia",["cultural","city","rural"],["culture","history","food","photography","nature"],"high",180,["spring","autumn"],True,.75),
    ("Interlaken","Switzerland","Europe",["mountain","adventure","nature"],["adventure","nature","photography","relaxation"],"high",220,["summer","winter"],True,.70),
    ("Paris","France","Europe",["city","cultural"],["culture","history","food","shopping","nightlife","photography"],"high",200,["spring","summer","autumn"],True,.95),
    ("Rome","Italy","Europe",["city","cultural"],["history","culture","food","photography"],"medium",160,["spring","autumn"],True,.85),
    ("Barcelona","Spain","Europe",["city","beach","cultural"],["culture","food","nightlife","photography","relaxation"],"medium",150,["spring","summer","autumn"],True,.90),
    ("Reykjavik","Iceland","Europe",["nature","adventure"],["nature","adventure","photography","relaxation"],"high",230,["summer","winter"],True,.65),
    ("Queenstown","New Zealand","Oceania",["mountain","adventure","nature"],["adventure","nature","photography"],"high",190,["summer","winter"],True,.60),
    ("Marrakech","Morocco","Africa",["desert","cultural","city"],["culture","history","food","shopping","adventure"],"low",80,["spring","autumn","winter"],True,.60),
    ("Bangkok","Thailand","Asia",["city","cultural"],["food","nightlife","culture","shopping","history"],"low",70,["winter","spring"],True,.85),
    ("Phuket","Thailand","Asia",["beach","island"],["relaxation","nightlife","adventure","food"],"medium",100,["winter","spring"],True,.80),
    ("Santorini","Greece","Europe",["island","coastal"],["relaxation","photography","food"],"high",210,["spring","summer","autumn"],True,.75),
    ("Cape Town","South Africa","Africa",["city","nature","adventure"],["nature","adventure","wildlife","food","photography"],"medium",120,["summer","autumn"],True,.70),
    ("Serengeti","Tanzania","Africa",["wildlife","nature","rural"],["wildlife","nature","photography","adventure"],"high",280,["summer","autumn"],False,.55),
    ("Cusco & Machu Picchu","Peru","South America",["mountain","cultural","adventure"],["history","culture","adventure","nature","photography"],"medium",110,["summer","autumn"],True,.65),
    ("Dubai","UAE","Asia",["city","luxury","desert"],["shopping","nightlife","relaxation","food","adventure"],"high",240,["winter","spring"],True,.80),
    ("Tokyo","Japan","Asia",["city","cultural"],["food","culture","shopping","nightlife","photography"],"high",190,["spring","autumn","winter"],True,.90),
    ("New York City","USA","North America",["city","cultural"],["culture","food","shopping","nightlife","photography","history"],"high",230,["spring","summer","autumn","winter"],True,.95),
    ("La Fortuna","Costa Rica","North America",["nature","adventure","rural"],["nature","adventure","wildlife","relaxation","photography"],"medium",130,["winter","spring"],True,.60),
    ("Amalfi Coast","Italy","Europe",["coastal","nature"],["relaxation","food","photography","adventure"],"high",220,["summer"],True,.70),
    ("Banff","Canada","North America",["mountain","nature","adventure"],["nature","adventure","photography","wildlife","relaxation"],"medium",150,["summer","winter"],True,.70),
    ("Prague","Czech Republic","Europe",["city","cultural"],["history","culture","nightlife","food","photography"],"medium",110,["spring","summer","autumn"],True,.75),
    ("Goa","India","Asia",["beach","coastal"],["relaxation","nightlife","food","adventure"],"low",60,["winter"],True,.70),
    ("Istanbul","Turkey","Europe",["city","cultural"],["history","culture","food","shopping","photography"],"medium",100,["spring","autumn"],True,.80),
    ("Maldives","Maldives","Asia",["island","beach","luxury"],["relaxation","adventure","photography"],"high",350,["winter","spring"],False,.75),
    ("Hanoi","Vietnam","Asia",["city","cultural","rural"],["food","culture","nightlife","adventure","shopping"],"low",55,["winter","spring"],True,.65),
]

# ------------------------------------------------------------------- helpers
def _wchoice(rng, dist):
    items = list(dist.keys())
    p = np.array([dist[k] for k in items], dtype=float)
    p /= p.sum()
    return str(rng.choice(items, p=p))

def _sample_interests(rng, dist, n):
    p = np.array([dist.get(i, 0.0) for i in INTERESTS], dtype=float)
    p /= p.sum()
    return [str(x) for x in rng.choice(INTERESTS, size=n, replace=False, p=p)]

def _assign_activities(rng, interests, max_acts=7):
    acts = []
    for it in interests:
        pool = ACTIVITY_POOLS.get(it, [])
        for a in rng.choice(pool, size=min(2, len(pool)), replace=False):
            if str(a) not in acts:
                acts.append(str(a))
    return acts[:max_acts]

def _compatibility(t, d, affinity):
    """Latent compatibility of traveler t (namedtuple row) with destination dict d."""
    ti, di = set(t.interests), set(d["interests"])
    inter = len(ti & di) / max(len(ti), 1)
    aff = float(np.mean([affinity.get(tag, 0.2) for tag in d["tags"]]))
    cost = d["avg_daily_cost"]
    budget_fit = 1.0 if cost <= t.budget_per_day else \
        max(0.0, 1 - (cost - t.budget_per_day) / max(t.budget_per_day, 1.0))
    season_fit = 1.0 if t.preferred_season in d["best_seasons"] else 0.4
    family_fit = 1.0 if (not t.with_children or d["family_friendly"]) else 0.3
    return 0.40 * inter + 0.25 * aff + 0.20 * budget_fit + 0.10 * season_fit + 0.05 * family_fit

# ------------------------------------------------------------- generators
def generate_travelers(n=N_TRAVELERS, seed=RANDOM_STATE):
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n):
        ttype = TRAVELER_TYPES[int(rng.integers(0, len(TRAVELER_TYPES)))]
        p = TYPE_PROFILES[ttype]
        age = int(np.clip(rng.normal(*p["age"]), 18, 75))
        income = float(max(800, rng.normal(*p["income"]) + 45 * (age - 25)))
        if age < 24 and rng.random() < 0.65:
            occupation = "student"
        elif age >= 60 and rng.random() < 0.55:
            occupation = "retired"
        else:
            occupation = _wchoice(rng, {"employed": .70, "self-employed": .20, "unemployed": .10})
        with_children = bool(rng.random() < p["children_prob"])
        gsize = int(max(1, round(rng.normal(*p["group_size"]))))
        if with_children:
            gsize = max(3, gsize)
        rows.append({
            "age": age,
            "gender": _wchoice(rng, {"M": .48, "F": .48, "male": .02, "F ": .02}),
            "occupation": occupation, "traveler_type": ttype,
            "monthly_income": round(income, 0),
            "budget_per_day": round(float(max(25, rng.normal(*p["budget"]))), 0),
            "trip_duration_days": int(np.clip(round(rng.normal(*p["duration"])), 2, 21)),
            "trips_per_year": round(float(max(0.5, rng.normal(*p["trips_per_year"]))), 1),
            "group_size": gsize, "with_children": with_children,
            "interests": _sample_interests(rng, p["interests"],
                                           int(np.clip(round(rng.normal(*p["n_interests"])), 2, 6))),
            "preferred_season": _wchoice(rng, p["season"]),
            "transport_pref": _wchoice(rng, p["transport"]),
            "accommodation_pref": _wchoice(rng, p["accommodation"]),
        })
    df = pd.DataFrame(rows)
    df.insert(0, "traveler_id", [f"T{i:04d}" for i in range(1, n + 1)])
    return df

def generate_destinations(seed=RANDOM_STATE + 7):
    rng = np.random.default_rng(seed)
    rows = []
    for i, (name, country, region, tags, interests, tier, cost, seasons, fam, pop) in enumerate(_DEST, 1):
        rows.append({"destination_id": f"D{i:03d}", "name": name, "country": country,
                     "region": region, "tags": list(tags), "interests": list(interests),
                     "budget_tier": tier, "avg_daily_cost": float(cost),
                     "best_seasons": list(seasons), "family_friendly": bool(fam),
                     "popularity": float(pop), "activities": _assign_activities(rng, interests)})
    return pd.DataFrame(rows)

def generate_ratings(travelers, destinations, seed=RANDOM_STATE + 13):
    rng = np.random.default_rng(seed)
    affinities = {t: p["tag_affinity"] for t, p in TYPE_PROFILES.items()}
    d_recs = destinations.to_dict("records")
    rows = []
    for t in travelers.itertuples(index=False):
        aff = affinities[t.traveler_type]
        comps = np.array([_compatibility(t, d, aff) for d in d_recs])
        p = comps ** 2
        p /= p.sum()
        idx = rng.choice(len(d_recs), size=min(len(d_recs), 3 + rng.poisson(9)),
                         replace=False, p=p)
        for j in idx:
            base = 1 + 4 * comps[j] + rng.normal(0, 0.35)
            rows.append({"traveler_id": t.traveler_id,
                         "destination_id": d_recs[j]["destination_id"],
                         "rating": int(np.clip(round(base), 1, 5))})
    return pd.DataFrame(rows)

def corrupt(df, seed=RANDOM_STATE + 21):
    """Inject realistic data-quality problems for the preprocessing stage."""
    df = df.copy()
    rng = np.random.default_rng(seed)
    n = len(df)
    def poke(col, frac):
        df.loc[rng.choice(n, size=int(frac * n), replace=False), col] = np.nan
    poke("monthly_income", 0.04); poke("budget_per_day", 0.03)
    poke("trip_duration_days", 0.02); poke("preferred_season", 0.02)
    poke("transport_pref", 0.02)
    df.loc[rng.choice(n, 30, replace=False), "gender"] = rng.choice([" male", "FEMALE "], 30)
    idx = rng.choice(n, size=max(3, int(0.01 * n)), replace=False)   # income outliers
    df.loc[idx, "monthly_income"] *= rng.uniform(6, 10, size=len(idx))
    return df

# ------------------------------------------------------------ load / save
def _split(s):
    return [x for x in str(s).split(";") if x and x != "nan"]

def load_or_generate(force=False):
    tp, dp, rp = DATA_DIR / "travelers.csv", DATA_DIR / "destinations.csv", DATA_DIR / "ratings.csv"
    if not force and tp.exists() and dp.exists() and rp.exists():
        t = pd.read_csv(tp, converters={"interests": _split})
        d = pd.read_csv(dp, converters={c: _split for c in
                        ["tags", "interests", "best_seasons", "activities"]})
        r = pd.read_csv(rp)
        return t, d, r
    travelers = generate_travelers()
    destinations = generate_destinations()
    ratings = generate_ratings(travelers, destinations)
    travelers = corrupt(travelers)
    t = travelers.copy(); t["interests"] = t["interests"].apply(";".join)
    d = destinations.copy()
    for c in ["tags", "interests", "best_seasons", "activities"]:
        d[c] = d[c].apply(";".join)
    t.to_csv(tp, index=False); d.to_csv(dp, index=False); ratings.to_csv(rp, index=False)
    return travelers, destinations, ratings