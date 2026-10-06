"""Stage 1: build one student-level dataset for the UTS MDSI competition.

Every row in the result represents exactly one student. This script does not
fit a model: it only loads, aggregates, merges, checks, and saves the data.
"""

from pathlib import Path

import pandas as pd


DATA_DIR = Path(__file__).parent / "uts-mdsi-competition-2026"
OUTPUT_PATH = DATA_DIR / "combined_students.csv"


def aggregate_attractions(attractions: pd.DataFrame) -> pd.DataFrame:
    """Turn many attraction visits per student into useful student summaries."""
    return (
        attractions.groupby("student_id", as_index=False)
        .agg(
            attraction_types_visited=("attraction_name", "nunique"),
            attraction_total_visits=("visit_count", "sum"),
            attraction_mean_visits=("visit_count", "mean"),
            attraction_max_visits=("visit_count", "max"),
        )
    )


def aggregate_monthly_activity(activity: pd.DataFrame) -> pd.DataFrame:
    """Turn monthly records into totals, averages, and an activity duration."""
    activity = activity.copy()
    activity["month_date"] = pd.to_datetime(activity["month_date"])

    monthly_summary = (
        activity.groupby("student_id", as_index=False)
        .agg(
            months_observed=("month_date", "nunique"),
            first_activity_month=("month_date", "min"),
            last_activity_month=("month_date", "max"),
            total_study_hours=("study_hours", "sum"),
            average_study_hours=("study_hours", "mean"),
            total_networking_events=("networking_events", "sum"),
            total_club_events=("club_events", "sum"),
            total_job_applications=("job_applications", "sum"),
            average_wellbeing_score=("wellbeing_score", "mean"),
            total_beach_visits=("beach_visits", "sum"),
            total_cultural_events=("cultural_events", "sum"),
            total_internship_hours=("internship_hours", "sum"),
        )
    )
    monthly_summary["activity_span_days"] = (
        monthly_summary["last_activity_month"]
        - monthly_summary["first_activity_month"]
    ).dt.days
    return monthly_summary.drop(columns=["first_activity_month", "last_activity_month"])


def require_one_row_per_student(data: pd.DataFrame, table_name: str) -> None:
    """Stop early if a merge table would multiply student rows."""
    duplicates = data["student_id"].duplicated().sum()
    if duplicates:
        raise ValueError(f"{table_name} has {duplicates} duplicate student IDs.")


def main() -> None:
    students = pd.read_csv(DATA_DIR / "students.csv")
    profiles = pd.read_csv(DATA_DIR / "student_profile_metrics.csv")
    suburbs = pd.read_csv(DATA_DIR / "suburbs.csv")
    attractions = pd.read_csv(DATA_DIR / "student_attractions.csv")
    activity = pd.read_csv(DATA_DIR / "student_monthly_activity.csv")
    labels = pd.read_csv(DATA_DIR / "train.csv")

    # These first three files already have one row per student or suburb.
    require_one_row_per_student(students, "students")
    require_one_row_per_student(profiles, "student_profile_metrics")

    attraction_summary = aggregate_attractions(attractions)
    activity_summary = aggregate_monthly_activity(activity)
    require_one_row_per_student(attraction_summary, "attraction summary")
    require_one_row_per_student(activity_summary, "monthly activity summary")

    # `validate` verifies the intended relationship for every merge.
    combined = students.merge(
        profiles, on="student_id", how="left", validate="one_to_one"
    )
    combined = combined.merge(
        suburbs, on="suburb_id", how="left", validate="many_to_one"
    )
    combined = combined.merge(
        attraction_summary, on="student_id", how="left", validate="one_to_one"
    )
    combined = combined.merge(
        activity_summary, on="student_id", how="left", validate="one_to_one"
    )
    combined = combined.merge(
        labels, on="student_id", how="left", validate="one_to_one"
    )

    require_one_row_per_student(combined, "combined dataset")
    print(f"Students in combined dataset: {len(combined):,}")
    print(f"Columns: {combined.shape[1]}")
    print(f"Labelled training students: {combined['successful_outcome'].notna().sum():,}")
    print(f"Unlabelled test students: {combined['successful_outcome'].isna().sum():,}")
    print("\nLargest missing-value counts:")
    print(combined.isna().sum().sort_values(ascending=False).head(15))

    combined.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
