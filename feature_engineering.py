"""Stage 2: create interpretable student-level features.

Input:  combined_students.csv from Code.py
Output: model_ready_students.csv

This script preserves every original column and adds derived columns. It does
not impute missing values, encode categories, scale values, or fit a model.
Those operations belong inside the later train/validation pipeline.
"""

from pathlib import Path

import numpy as np
import pandas as pd


DATA_DIR = Path(__file__).parent / "uts-mdsi-competition-2026"
INPUT_PATH = DATA_DIR / "combined_students.csv"
OUTPUT_PATH = DATA_DIR / "model_ready_students.csv"


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Divide without turning zero denominators into infinite feature values."""
    return numerator / denominator.replace(0, np.nan)


def add_activity_trends(data: pd.DataFrame) -> pd.DataFrame:
    """Create first-to-last monthly changes for measures where trend matters."""
    activity = pd.read_csv(DATA_DIR / "student_monthly_activity.csv")
    activity["month_date"] = pd.to_datetime(activity["month_date"])
    activity = activity.sort_values(["student_id", "month_date"])
    tracked_columns = [
        "study_hours",
        "wellbeing_score",
        "networking_events",
        "job_applications",
        "internship_hours",
    ]
    first = activity.groupby("student_id")[tracked_columns].first()
    last = activity.groupby("student_id")[tracked_columns].last()
    changes = (last - first).add_prefix("change_").reset_index()
    changes = changes.rename(
        columns={f"change_{name}": f"first_to_last_change_{name}" for name in tracked_columns}
    )
    result = data.merge(changes, on="student_id", how="left", validate="one_to_one")
    elapsed_months = result["activity_span_days"] / 30.4375
    for name in tracked_columns:
        result[f"monthly_change_{name}"] = safe_divide(
            result[f"first_to_last_change_{name}"], elapsed_months
        )
    return result


def add_attraction_concentration(data: pd.DataFrame) -> pd.DataFrame:
    """Show whether visits are spread across attractions or focused on one."""
    attractions = pd.read_csv(DATA_DIR / "student_attractions.csv")
    summary = attractions.groupby("student_id")["visit_count"].agg(["sum", "max"])
    summary["attraction_top_visit_share"] = safe_divide(summary["max"], summary["sum"])
    summary["attraction_diversity_ratio"] = safe_divide(
        data.set_index("student_id")["attraction_types_visited"].reindex(summary.index),
        summary["sum"],
    )
    summary = summary[["attraction_top_visit_share", "attraction_diversity_ratio"]].reset_index()
    return data.merge(summary, on="student_id", how="left", validate="one_to_one")


def add_features(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    data = add_activity_trends(data)
    data = add_attraction_concentration(data)

    # Rates use months observed so students with different observation periods
    # can be compared fairly.
    months = data["months_observed"]
    for column in [
        "total_study_hours",
        "total_networking_events",
        "total_club_events",
        "total_job_applications",
        "total_beach_visits",
        "total_cultural_events",
        "total_internship_hours",
    ]:
        feature_name = column.removeprefix("total_") + "_per_month"
        data[feature_name] = safe_divide(data[column], months)

    # Direct measures of academic engagement and professional progress.
    data["academic_participation_average"] = (
        data["attendance_rate"] + data["assignment_submission_rate"]
        + data["group_project_participation"] + data["workshop_attendance"]
    ) / 4
    data["assignment_attendance_gap"] = (
        data["assignment_submission_rate"] - data["attendance_rate"]
    )
    data["internship_completion_rate"] = safe_divide(
        data["internships_completed"], data["internship_applications"]
    )
    data["internship_hours_per_completed"] = safe_divide(
        data["total_internship_hours"], data["internships_completed"]
    )
    data["mentor_connections_per_linkedin_connection"] = safe_divide(
        data["industry_mentor_sessions"], data["linkedin_connections"]
    )

    # Campus, community, and career participation.
    data["events_per_month"] = safe_divide(
        data["total_networking_events"] + data["total_club_events"]
        + data["total_cultural_events"],
        months,
    )
    data["career_actions_per_month"] = safe_divide(
        data["total_job_applications"] + data["internship_applications"], months
    )
    data["attraction_visits_per_type"] = safe_divide(
        data["attraction_total_visits"], data["attraction_types_visited"]
    )

    # Time and travel features. These are proxies to test, not assumptions
    # about any individual student's circumstances.
    data["weekly_screen_hours"] = (
        data["weekly_streaming_hours"] + data["gaming_hours"]
    )
    data["exercise_per_week"] = data["exercise_sessions_per_month"] / 4.345
    data["commute_minutes_per_km"] = safe_divide(
        data["commute_time_minutes"], data["distance_to_uts_km"]
    )
    data["work_to_study_ratio"] = safe_divide(
        data["part_time_work_hours"] * 4.345, data["average_study_hours"]
    )

    # Log features reduce the impact of a few unusually large count values.
    # We retain the original columns too, so validation can decide their value.
    for column in [
        "linkedin_connections",
        "attraction_total_visits",
        "total_job_applications",
        "total_internship_hours",
        "total_study_hours",
    ]:
        data[f"log1p_{column}"] = np.log1p(data[column].clip(lower=0))

    return data.replace([np.inf, -np.inf], np.nan)


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"{INPUT_PATH} was not found. Run Code.py to build the merged dataset first."
        )

    combined = pd.read_csv(INPUT_PATH)
    if combined["student_id"].duplicated().any():
        raise ValueError("Input must contain exactly one row per student.")

    model_ready = add_features(combined)
    new_columns = [column for column in model_ready if column not in combined]
    print(f"Input shape: {combined.shape}")
    print(f"Output shape: {model_ready.shape}")
    print(f"New features ({len(new_columns)}):")
    print("\n".join(new_columns))
    print("\nMissing values in new features:")
    print(model_ready[new_columns].isna().sum().sort_values(ascending=False))

    model_ready.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
