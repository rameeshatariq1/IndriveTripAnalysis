import pandas as pd
import numpy as np
import os

TRIPS_RAW   = "data/trips.csv"
SURVEY_RAW  = "data/survey.csv"
TRIPS_CLEAN = "data/trips_clean.csv"
SURVEY_CLEAN= "data/survey_clean.csv"

def clean_trips(path: str = TRIPS_RAW) -> pd.DataFrame:
    df = pd.read_csv(path)

    print(f"Raw shape: {df.shape}")
    print(f"Missing values:\n{df.isnull().sum()}\n")

    # 1. Normalize Ride Type casing  (Car / car / CAR → Car)
    df["Ride Type"] = df["Ride Type"].str.strip().str.capitalize()

    # 2. Normalize area names (strip whitespace)
    for col in ["Pickup Area", "Dropoff Area"]:
        df[col] = df[col].str.strip().str.title()

    # 6. Feature engineering
    df["Negotiation_Gap"]     = df["Final Accepted Fare (PKR)"] - df["Suggested Fare (PKR)"]
    df["Negotiation_Gap_Pct"] = (df["Negotiation_Gap"] / df["Suggested Fare (PKR)"]) * 100

    # 7. Ordered time category for sorting in charts
    time_order = [
        "Early Morning (5–8 AM)",
        "Morning (8–11 AM)",
        "Afternoon (11 AM–2 PM)",
        "Evening (2–6 PM)",
        "Night (6–10 PM)",
        "Late Night (10 PM+)",
    ]
    df["Time Category"] = pd.Categorical(
        df["Time Category"], categories=time_order, ordered=True
    )

    print(f"\nClean shape: {df.shape}")
    print(f"Ride type counts:\n{df['Ride Type'].value_counts()}")
    return df

SURVEY_RENAME = {
    "SECTION 1: Basic Info\nWhat is your gender?"                                : "Gender",
    "What is your age group?"                                                     : "Age_Group",
    "What is your occupation?"                                                    : "Occupation",
    "How often do you use inDrive?"                                               : "Usage_Frequency",
    "SECTION 2: Trip Details \nWhich area do you usually travel FROM?"            : "Travel_From",
    "Which area do you usually travel TO? "                                       : "Travel_To",
    "What is your most common reason for using inDrive?"                          : "Trip_Reason",
    "What time do you usually book rides? (Select all that apply)"                : "Booking_Time",
    "SECTION 3: Fare & Negotiation \nWhat fare do you usually offer for your most common route? (in PKR)": "Fare_Offered_PKR",
    "Does inDrive's suggested fare feel reasonable to you?"                       : "Suggested_Fare_Reasonable",
    "How much do you typically offer compared to the suggested fare?"             : "Offer_vs_Suggested",
    "How many drivers usually respond to your request?"                           : "Driver_Responses",
    "Do you wait longer to get a ride during certain times?"                      : "Wait_Longer",
    "SECTION 4: Peak Hours & Pricing Perception \nHave you noticed fares going higher at certain times of day?": "Fares_Higher_Noticed",
    "If yes, when do you feel fares are highest? (Select all that apply)"         : "Fares_Highest_When",
    "Do you change your travel time to avoid high fares?"                         : "Change_Travel_Time",
    "Has rain or bad weather affected your fare or wait time?"                    : "Weather_Affected",
    "SECTION 5: Overall Experience \nCompared to other ride-hailing apps in Lahore (Careem, Uber, Bykea), inDrive fares are:": "Vs_Other_Apps",
    "What is your biggest frustration with inDrive?"                              : "Biggest_Frustration",
    "Would you recommend inDrive to others in Lahore?"                            : "Would_Recommend",
}

def clean_survey(path: str = SURVEY_RAW) -> pd.DataFrame:
    sv = pd.read_csv(path)
    rename_map = {}
    for col in sv.columns:
        for key, val in SURVEY_RENAME.items():
            if key.strip() in col.strip() or col.strip() in key.strip():
                rename_map[col] = val
                break

    sv = sv.rename(columns=rename_map)
    sv["Timestamp"] = pd.to_datetime(sv["Timestamp"], errors="coerce")

    # Convert fare offered to numeric
    if "Fare_Offered_PKR" in sv.columns:
        sv["Fare_Offered_PKR"] = pd.to_numeric(sv["Fare_Offered_PKR"], errors="coerce")

    print(f"Survey clean shape: {sv.shape}")
    print(f"Columns: {sv.columns.tolist()}")
    return sv

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)

    trips  = clean_trips()
    survey = clean_survey()

    trips.to_csv(TRIPS_CLEAN,  index=False)
    survey.to_csv(SURVEY_CLEAN, index=False)

    print("\nCleaned files saved to data/trips_clean.csv and data/survey_clean.csv")