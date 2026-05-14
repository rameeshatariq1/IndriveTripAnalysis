import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import os

from data_cleaning import clean_trips, clean_survey

os.makedirs("output/plots", exist_ok=True)

PURPLE = "#7F77DD"
TEAL   = "#1D9E75"
CORAL  = "#D85A30"
AMBER  = "#EF9F27"
GRAY   = "#888780"
BLUE   = "#378ADD"

sns.set_theme(style="whitegrid", font_scale=1.05)
plt.rcParams.update({"figure.dpi": 120, "axes.spines.top": False,
                      "axes.spines.right": False})


def save(fig, name: str):
    path = f"output/plots/{name}.png"
    fig.savefig(path, bbox_inches="tight")
    print(f"  Saved → {path}")
    plt.close(fig)

def print_stats(df: pd.DataFrame):
    print("\n========== DESCRIPTIVE STATISTICS ==========")
    print(df[["Suggested Fare (PKR)", "Final Accepted Fare (PKR)",
              "Num of Driver Bids", "Negotiation_Gap_Pct"]].describe().round(2))
    print(f"\nAvg Negotiation Gap: ₨{df['Negotiation_Gap'].mean():.1f}")
    print(f"Avg Gap %:           {df['Negotiation_Gap_Pct'].mean():.1f}%")
    print(f"Trips where final > suggested: "
          f"{(df['Final Accepted Fare (PKR)'] > df['Suggested Fare (PKR)']).sum()} "
          f"/ {len(df)}")
    
def plot_ride_distribution(df: pd.DataFrame):
    counts = df["Ride Type"].value_counts()
    colors = [PURPLE, TEAL, CORAL]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # Donut
    wedges, texts, autotexts = axes[0].pie(
        counts, labels=counts.index, autopct="%1.1f%%",
        colors=colors, startangle=90,
        wedgeprops={"width": 0.55, "edgecolor": "white", "linewidth": 2}
    )
    for t in autotexts: t.set_fontsize(10)
    axes[0].set_title("Ride Type Distribution", fontweight="bold")

    # Bar
    bars = axes[1].bar(counts.index, counts.values, color=colors, width=0.55, edgecolor="white")
    axes[1].bar_label(bars, padding=3)
    axes[1].set_ylabel("Number of Trips")
    axes[1].set_title("Trip Count per Ride Type", fontweight="bold")

    fig.suptitle("Fig 1 — Ride Type Analysis (n=498)", y=1.02, fontsize=11, color=GRAY)
    save(fig, "01_ride_type_distribution")

def plot_time_analysis(df: pd.DataFrame):
    time_counts = df["Time Category"].value_counts().sort_index()
    time_fare   = df.groupby("Time Category", observed=True)["Final Accepted Fare (PKR)"].mean()

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    # Volume
    axes[0].bar(time_counts.index, time_counts.values, color=PURPLE, edgecolor="white")
    axes[0].set_xticklabels(time_counts.index, rotation=35, ha="right", fontsize=9)
    axes[0].set_ylabel("Trip Count")
    axes[0].set_title("Trip Volume by Time of Day", fontweight="bold")
    for i, v in enumerate(time_counts.values):
        axes[0].text(i, v + 1, str(v), ha="center", fontsize=9)

    # Avg fare
    colors_fare = [TEAL if v == time_fare.max() else PURPLE for v in time_fare.values]
    axes[1].bar(time_fare.index, time_fare.values, color=colors_fare, edgecolor="white")
    axes[1].set_xticklabels(time_fare.index, rotation=35, ha="right", fontsize=9)
    axes[1].yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"₨{x:.0f}"))
    axes[1].set_ylabel("Avg Final Fare (PKR)")
    axes[1].set_title("Avg Accepted Fare by Time (highlighted = highest)", fontweight="bold")
    for i, v in enumerate(time_fare.values):
        axes[1].text(i, v + 8, f"₨{v:.0f}", ha="center", fontsize=9)

    fig.suptitle("Fig 2 — Time-Based Analysis: Night hours = most trips, Early Morning = highest fares",
                 y=1.02, fontsize=10, color=GRAY)
    save(fig, "02_time_analysis")

def plot_negotiation_gap(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Scatter: suggested vs final
    colors_map = {"Car": PURPLE, "Bike": TEAL, "Rickshaw": CORAL}
    for ride, grp in df.groupby("Ride Type"):
        axes[0].scatter(grp["Suggested Fare (PKR)"], grp["Final Accepted Fare (PKR)"],
                        alpha=0.45, s=30, label=ride, color=colors_map.get(ride, GRAY))
    max_val = df[["Suggested Fare (PKR)", "Final Accepted Fare (PKR)"]].max().max()
    axes[0].plot([0, max_val], [0, max_val], "k--", linewidth=1, alpha=0.4, label="Suggested = Final")
    axes[0].set_xlabel("Suggested Fare (PKR)")
    axes[0].set_ylabel("Final Accepted Fare (PKR)")
    axes[0].set_title("Suggested vs Final Fare\n(above dashed line = overpayment)", fontweight="bold")
    axes[0].legend(fontsize=9)

    gap_by_type = df.groupby("Ride Type")["Negotiation_Gap_Pct"].mean().sort_values()
    bar_colors  = [colors_map.get(r, GRAY) for r in gap_by_type.index]
    bars = axes[1].barh(gap_by_type.index, gap_by_type.values, color=bar_colors, edgecolor="white")
    axes[1].axvline(0, color="black", linewidth=0.8)
    axes[1].set_xlabel("Avg Negotiation Gap (%)")
    axes[1].set_title("Avg Overpayment % by Ride Type\n(negative = user paid MORE than suggested)",
                      fontweight="bold")
    for bar, v in zip(bars, gap_by_type.values):
        axes[1].text(v - 1.5, bar.get_y() + bar.get_height() / 2,
                     f"{v:.1f}%", va="center", fontsize=10, color="white", fontweight="bold")

    fig.suptitle("Fig 3 — Negotiation Gap: Riders pay 11–15% above inDrive's own suggestion",
                 y=1.02, fontsize=10, color=GRAY)
    save(fig, "03_negotiation_gap")

def plot_driver_bids(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Bids by ride type (boxplot)
    ride_order = ["Car", "Bike", "Rickshaw"]
    palette    = {"Car": PURPLE, "Bike": TEAL, "Rickshaw": CORAL}
    sns.boxplot(data=df, x="Ride Type", y="Num of Driver Bids",
                order=ride_order, palette=palette, ax=axes[0], width=0.5)
    axes[0].set_title("Driver Bid Distribution by Ride Type", fontweight="bold")
    axes[0].set_ylabel("Number of Driver Bids")

    # Bids vs fare (scatter)
    axes[1].scatter(df["Num of Driver Bids"], df["Final Accepted Fare (PKR)"],
                    alpha=0.35, s=28, color=BLUE)
    # trend line
    z = np.polyfit(df["Num of Driver Bids"].dropna(),
                   df.loc[df["Num of Driver Bids"].notna(), "Final Accepted Fare (PKR)"], 1)
    xline = np.linspace(df["Num of Driver Bids"].min(), df["Num of Driver Bids"].max(), 100)
    axes[1].plot(xline, np.poly1d(z)(xline), color=CORAL, linewidth=2, label="Trend")
    axes[1].set_xlabel("Number of Driver Bids")
    axes[1].set_ylabel("Final Accepted Fare (PKR)")
    axes[1].set_title("Driver Bids vs Final Fare\n(more competition → lower fare?)", fontweight="bold")
    axes[1].legend()

    fig.suptitle("Fig 4 — Driver Supply: Cars attract most competition, Rickshaws least",
                 y=1.02, fontsize=10, color=GRAY)
    save(fig, "04_driver_bids")

def plot_environmental_impact(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Fare by weather
    weather_fare = df.groupby("Weather Condition")["Final Accepted Fare (PKR)"].mean().sort_values(ascending=False)
    colors_w = [CORAL if w == "Rain" else (AMBER if w == "Cloudy" else TEAL) for w in weather_fare.index]
    bars = axes[0].bar(weather_fare.index, weather_fare.values, color=colors_w, edgecolor="white", width=0.5)
    axes[0].bar_label(bars, fmt="₨%.0f", padding=3)
    axes[0].set_ylabel("Avg Final Fare (PKR)")
    axes[0].set_title("Avg Fare by Weather Condition\n(Rain = highest fares)", fontweight="bold")

    # Fare by traffic
    traffic_fare = df.groupby("Traffic Level")["Final Accepted Fare (PKR)"].mean()
    bars2 = axes[1].bar(traffic_fare.index, traffic_fare.values,
                        color=[CORAL, TEAL], edgecolor="white", width=0.5)
    axes[1].bar_label(bars2, fmt="₨%.0f", padding=3)
    axes[1].set_ylabel("Avg Final Fare (PKR)")
    axes[1].set_title("Avg Fare by Traffic Level\n(Moderate traffic → higher fares)", fontweight="bold")

    fig.suptitle("Fig 5 — Environmental Penalty: Rain & traffic raise fares significantly",
                 y=1.02, fontsize=10, color=GRAY)
    save(fig, "05_environmental_impact")

def plot_top_areas(df: pd.DataFrame):
    top10 = df["Pickup Area"].value_counts().head(10).sort_values()

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(top10.index, top10.values, color=PURPLE, edgecolor="white")
    ax.bar_label(bars, padding=3)
    ax.set_xlabel("Number of Trips")
    ax.set_title("Fig 6 — Top 10 Pickup Areas in Lahore\n(Gulberg belt dominates inDrive demand)",
                 fontweight="bold")
    save(fig, "06_top_pickup_areas")

if __name__ == "__main__":
    df     = clean_trips()
    survey = clean_survey()

    print_stats(df)
    print("\nGenerating plots...")
    plot_ride_distribution(df)
    plot_time_analysis(df)
    plot_negotiation_gap(df)
    plot_driver_bids(df)
    plot_environmental_impact(df)
    plot_top_areas(df)

    print("\nAll 7 plots saved to output/plots/")