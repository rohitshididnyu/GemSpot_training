from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


RNG = np.random.default_rng(42)

CATEGORIES = ["park", "museum", "restaurant", "beach", "hike", "market"]
REGIONS = ["hudson_valley", "brooklyn", "queens", "catskills", "long_island", "jersey_shore"]
VIBE_TAGS = ["scenic", "relaxing", "foodie", "nightlife"]
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
SERVICES_BY_CATEGORY = {
    "park": ["trails", "picnic_area", "lake_view", "parking"],
    "museum": ["guided_tours", "gift_shop", "family_tickets", "parking"],
    "restaurant": ["reservations", "outdoor_seating", "delivery", "late_night"],
    "beach": ["boardwalk", "parking", "restrooms", "rentals"],
    "hike": ["trail_map", "viewpoints", "parking", "camping"],
    "market": ["food_stalls", "souvenirs", "weekend_only", "music"],
}
REVIEW_SNIPPETS = {
    "scenic": [
        "beautiful views and quiet atmosphere",
        "perfect for a peaceful day trip",
        "great scenery and easy to explore",
    ],
    "relaxing": [
        "calm vibe and not too crowded",
        "easy to spend a slow afternoon here",
        "felt peaceful and comfortable",
    ],
    "foodie": [
        "excellent food options and worth the drive",
        "lots of local flavors and hidden gems",
        "the food scene was the highlight",
    ],
    "nightlife": [
        "busy after dark with lots happening",
        "fun evening atmosphere and music",
        "best visited later in the day",
    ],
}


def sample_services(category: str) -> str:
    services = RNG.choice(SERVICES_BY_CATEGORY[category], size=3, replace=False)
    return "|".join(services.tolist())


def generate_row(index: int) -> dict:
    vibe = RNG.choice(VIBE_TAGS)
    category = RNG.choice(CATEGORIES)

    scenic_pref = float(np.clip(RNG.normal(0.55, 0.2), 0, 1))
    relaxing_pref = float(np.clip(RNG.normal(0.5, 0.2), 0, 1))
    foodie_pref = float(np.clip(RNG.normal(0.5, 0.2), 0, 1))
    nightlife_pref = float(np.clip(RNG.normal(0.45, 0.2), 0, 1))
    preference_lookup = {
        "scenic": scenic_pref,
        "relaxing": relaxing_pref,
        "foodie": foodie_pref,
        "nightlife": nightlife_pref,
    }

    avg_rating = float(np.clip(RNG.normal(4.1, 0.45), 2.5, 5.0))
    place_popularity = float(np.clip(RNG.normal(0.6, 0.2), 0, 1))
    distance_km = float(np.clip(RNG.normal(45, 22), 2, 140))
    total_visits_30d = int(np.clip(RNG.normal(10, 5), 0, 30))
    saved_places_30d = int(np.clip(RNG.normal(4, 3), 0, 15))
    num_reviews = int(np.clip(RNG.normal(220, 180), 5, 1200))
    month = int(RNG.integers(1, 13))
    day_of_week = str(RNG.choice(DAYS))

    vibe_alignment = preference_lookup[vibe]
    rating_score = (avg_rating - 3.0) / 2.0
    distance_score = 1.0 - min(distance_km / 140.0, 1.0)
    popularity_score = place_popularity
    activity_score = min(saved_places_30d / 10.0, 1.0)
    weekend_bonus = 0.1 if day_of_week in {"Friday", "Saturday", "Sunday"} else 0.0

    latent_score = (
        1.35 * vibe_alignment
        + 0.75 * rating_score
        + 0.65 * distance_score
        + 0.45 * popularity_score
        + 0.25 * activity_score
        + weekend_bonus
        + RNG.normal(0, 0.18)
    )
    will_visit = int(latent_score > 1.6)

    return {
        "user_id": f"user_{index % 250:04d}",
        "destination_id": f"dest_{index:05d}",
        "total_visits_30d": total_visits_30d,
        "saved_places_30d": saved_places_30d,
        "scenic_pref": scenic_pref,
        "relaxing_pref": relaxing_pref,
        "foodie_pref": foodie_pref,
        "nightlife_pref": nightlife_pref,
        "category": category,
        "region": str(RNG.choice(REGIONS)),
        "vibe_tag": vibe,
        "services": sample_services(category),
        "num_reviews": num_reviews,
        "place_popularity": place_popularity,
        "avg_rating": avg_rating,
        "distance_km": distance_km,
        "day_of_week": day_of_week,
        "month": month,
        "review_text": str(RNG.choice(REVIEW_SNIPPETS[vibe])),
        "will_visit": will_visit,
    }


def generate_frame(rows: int) -> pd.DataFrame:
    return pd.DataFrame([generate_row(index) for index in range(rows)])


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a synthetic GemSpot dataset.")
    parser.add_argument("--output-dir", default="data/demo", help="Directory for generated CSV files.")
    parser.add_argument("--train-rows", type=int, default=2500, help="Number of training rows.")
    parser.add_argument("--val-rows", type=int, default=600, help="Number of validation rows.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_frame = generate_frame(args.train_rows)
    val_frame = generate_frame(args.val_rows)

    train_path = output_dir / "gemspot_train.csv"
    val_path = output_dir / "gemspot_val.csv"

    train_frame.to_csv(train_path, index=False)
    val_frame.to_csv(val_path, index=False)

    print(f"Wrote {train_path}")
    print(f"Wrote {val_path}")


if __name__ == "__main__":
    main()
