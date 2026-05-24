import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def generate_sample_data(num_students: int = 150) -> pd.DataFrame:
    """Create a synthetic student dataset for analytics demonstration."""
    np.random.seed(42)
    student_ids = [f"S{1000 + i}" for i in range(num_students)]

    genders = np.random.choice(["Female", "Male", "Non-binary"], size=num_students, p=[0.48, 0.48, 0.04])
    grade_levels = np.random.choice([9, 10, 11, 12], size=num_students, p=[0.25, 0.25, 0.25, 0.25])
    socio_economic_status = np.random.choice(["Low", "Middle", "High"], size=num_students, p=[0.3, 0.5, 0.2])

    attendance_rate = np.clip(np.random.normal(loc=0.92, scale=0.06, size=num_students), 0.7, 1.0)
    avg_marks = np.clip(np.random.normal(loc=70, scale=12, size=num_students), 35, 100)
    behaviour_score = np.clip(np.random.normal(loc=7.4, scale=1.8, size=num_students), 1, 10)

    math_score = np.clip(np.random.normal(loc=avg_marks + 2, scale=10, size=num_students), 30, 100)
    reading_score = np.clip(np.random.normal(loc=avg_marks + 1, scale=9, size=num_students), 30, 100)
    science_score = np.clip(np.random.normal(loc=avg_marks - 1, scale=10, size=num_students), 30, 100)

    extracurricular = np.random.choice(["Sports", "Arts", "Science Club", "None", "Debate"], size=num_students, p=[0.18, 0.18, 0.18, 0.35, 0.11])
    attendance_days = np.clip(np.round(attendance_rate * 180).astype(int), 120, 180)

    data = pd.DataFrame(
        {
            "StudentID": student_ids,
            "Gender": genders,
            "Grade": grade_levels,
            "SES": socio_economic_status,
            "AttendanceRate": attendance_rate,
            "AttendanceDays": attendance_days,
            "MathScore": math_score,
            "ReadingScore": reading_score,
            "ScienceScore": science_score,
            "BehaviourScore": behaviour_score,
            "Extracurricular": extracurricular,
        }
    )
    data["AverageScore"] = data[["MathScore", "ReadingScore", "ScienceScore"]].mean(axis=1)
    data["PerformanceBand"] = pd.cut(
        data["AverageScore"],
        bins=[0, 50, 65, 80, 100],
        labels=["Poor", "Below Average", "Average", "Strong"],
        include_lowest=True,
    )
    return data


def load_data(csv_path: str = None) -> pd.DataFrame:
    """Load student data from CSV if it exists, otherwise generate sample data."""
    if csv_path and os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        return df

    print("No CSV file found or provided. Generating synthetic sample data.")
    return generate_sample_data()


def clean_and_engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize and derive analytics-ready features."""
    df = df.copy()
    if "AverageScore" not in df.columns:
        score_cols = [c for c in df.columns if c.endswith("Score") and c != "BehaviourScore"]
        df["AverageScore"] = df[score_cols].mean(axis=1)

    df["AttendanceRate"] = df["AttendanceRate"].fillna(df.get("AttendanceDays", 0) / 180)
    df["AtRisk"] = (
        (df["AverageScore"] < 55)
        | (df["AttendanceRate"] < 0.85)
        | (df.get("BehaviourScore", 10) < 5)
    )
    df["PerformanceCategory"] = pd.cut(
        df["AverageScore"],
        bins=[0, 55, 65, 75, 100],
        labels=["Critical", "Low", "Moderate", "High"],
        include_lowest=True,
    )
    return df


def score_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Generate summary statistics for numeric performance indicators."""
    summary = df[
        ["AttendanceRate", "MathScore", "ReadingScore", "ScienceScore", "AverageScore", "BehaviourScore"]
    ].describe().transpose()
    return summary


def demographic_analysis(df: pd.DataFrame):
    """Analyze student performance by demographic groups."""
    groups = ["Gender", "Grade", "SES", "Extracurricular"]
    demographic_metrics = []

    for group in groups:
        if group in df.columns:
            metrics = (
                df.groupby(group)
                .agg(
                    count=("StudentID", "count"),
                    avg_score=("AverageScore", "mean"),
                    attendance=("AttendanceRate", "mean"),
                    at_risk_pct=("AtRisk", lambda x: 100 * x.mean()),
                )
                .reset_index()
            )
            demographic_metrics.append((group, metrics))

    return demographic_metrics


def correlation_analysis(df: pd.DataFrame) -> pd.Series:
    """Identify the strongest relationships between variables."""
    numeric_cols = ["AttendanceRate", "MathScore", "ReadingScore", "ScienceScore", "AverageScore", "BehaviourScore"]
    corr_matrix = df[numeric_cols].corr()["AverageScore"].sort_values(ascending=False)
    return corr_matrix


def identify_at_risk_students(df: pd.DataFrame, top_n: int = 25) -> pd.DataFrame:
    """Flag and rank at-risk students based on attendance, scores, and behavior."""
    if "RiskScore" not in df.columns:
        df["RiskScore"] = (
            (60 - df["AverageScore"]).clip(lower=0) * 0.5
            + ((0.90 - df["AttendanceRate"]).clip(lower=0) * 100) * 0.35
            + ((5 - df.get("BehaviourScore", 10)).clip(lower=0) * 3) * 0.15
        )
    alerts = df.sort_values(by=["AtRisk", "RiskScore"], ascending=[False, False])
    return alerts.head(top_n)


def plot_performance_distribution(df: pd.DataFrame, output_folder: str = "reports") -> None:
    """Create visual dashboards showing performance distributions."""
    os.makedirs(output_folder, exist_ok=True)
    sns.set(style="whitegrid", palette="muted")

    plt.figure(figsize=(10, 6))
    sns.histplot(df["AverageScore"], bins=12, kde=True, color="#3c8dbc")
    plt.title("Average Score Distribution")
    plt.xlabel("Average Score")
    plt.ylabel("Number of Students")
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "average_score_distribution.png"))
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x="PerformanceCategory", order=["Critical", "Low", "Moderate", "High"])
    plt.title("Student Performance Category Counts")
    plt.xlabel("Performance Category")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "performance_category_counts.png"))
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=df,
        x="AttendanceRate",
        y="AverageScore",
        hue="PerformanceCategory",
        palette="viridis",
        alpha=0.8,
    )
    plt.title("Attendance Rate vs Average Score")
    plt.xlabel("Attendance Rate")
    plt.ylabel("Average Score")
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "attendance_vs_score.png"))
    plt.close()


def export_reports(df: pd.DataFrame, output_folder: str = "reports") -> None:
    """Write analytics reports and filtered student lists to CSV files."""
    os.makedirs(output_folder, exist_ok=True)
    df.to_csv(os.path.join(output_folder, "student_performance_full.csv"), index=False)
    df[df["AtRisk"]].to_csv(os.path.join(output_folder, "at_risk_students.csv"), index=False)

    summary_table = score_summary(df)
    summary_table.to_csv(os.path.join(output_folder, "performance_summary.csv"))

    print(f"Reports exported to {os.path.abspath(output_folder)}")


def display_insights(df: pd.DataFrame) -> None:
    """Print key analytics findings and recommendations."""
    print("\n=== Key Performance Insights ===")
    print(score_summary(df))
    print("\n=== Correlation with Average Performance ===")
    print(correlation_analysis(df))

    demographics = demographic_analysis(df)
    for group, metrics in demographics:
        print(f"\n=== Performance by {group} ===")
        print(metrics)

    at_risk_students = identify_at_risk_students(df, top_n=15)
    print("\n=== Top At-Risk Students ===")
    print(at_risk_students[["StudentID", "Grade", "AverageScore", "AttendanceRate", "BehaviourScore", "RiskScore", "AtRisk"]])

    num_at_risk = int(df["AtRisk"].sum())
    print(f"\nTotal At-Risk Students: {num_at_risk} / {len(df)} ({num_at_risk / len(df) * 100:.1f}%)")


def main(csv_path: str = None, output_folder: str = "reports") -> None:
    df = load_data(csv_path)
    df = clean_and_engineer_features(df)
    display_insights(df)
    export_reports(df, output_folder)
    plot_performance_distribution(df, output_folder)
    print("\nAnalytics complete. Open the reports folder to inspect CSVs and charts.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Student Performance Analytics System"
    )
    parser.add_argument(
        "--csv",
        dest="csv_path",
        default=None,
        help="Path to a CSV file containing student data.",
    )
    parser.add_argument(
        "--output",
        dest="output_folder",
        default="reports",
        help="Folder where analytics reports and charts will be saved.",
    )
    args = parser.parse_args()
    main(csv_path=args.csv_path, output_folder=args.output_folder)
