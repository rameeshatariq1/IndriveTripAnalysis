# inDrive Lahore — Trip Analysis
**Real-World Data Science Project | BS Artificial Intelligence | Superior University**

---

## Problem Statement
inDrive riders in Lahore suffer from **fare blindness** — no data to judge whether the app's suggested fare is fair, what time gives cheapest rides, or which vehicle gets the fastest driver response. This project analyzes 498 real trips + 42 user survey responses to surface actionable insights.

## 3 Problems Solved
| Problem | Finding |
|---|---|
| Fare Blindness | Riders overpay by 12.6% above app suggestion consistently |
| Time Trap | Night fares are 60% higher than morning fares |
| Ride Type Confusion | Cars get 3.94 avg bids vs rickshaws at 2.48 |

---

## Project Structure
```
inDrive_Lahore_Analysis/
├── data/
│   ├── trips.csv           # 498 manual trip logs
│   └── survey.csv          # 42 user survey responses
├── data_cleaning.py        # Step 1: Clean & preprocess both datasets
├── eda_analysis.py         # Step 2: Generate 7 EDA plots
├── model.py                # Step 3: Fare prediction model
├── app.py                  # Streamlit portfolio dashboard
├── requirements.txt
└── README.md
```

---

## How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Add your data files
Place your CSVs in the `data/` folder:
- `data/trips.csv` (main trip log)
- `data/survey.csv` (user survey)

### 3. Run data cleaning
```bash
python data_cleaning.py
```

### 4. Run EDA (generates plots)
```bash
python eda_analysis.py
```

### 5. Run model
```bash
python model.py
```

### 6. Launch Streamlit app (main deliverable)
```bash
streamlit run app.py
```

---

## Data Description

### trips.csv (498 rows)
| Column | Description |
|---|---|
| Date | Trip date |
| Team Member Name | Data collector |
| Time Category | Time slot (Morning/Evening/Night etc.) |
| Pickup Area | Origin zone in Lahore |
| Dropoff Area | Destination zone |
| Ride Type | Car / Bike / Rickshaw |
| Suggested Fare (PKR) | inDrive app suggestion |
| Team Member Offer (PKR) | Fare offered by rider |
| Final Accepted Fare (PKR) | Actual fare paid |
| Num of Driver Bids | Driver responses received |
| Weather Condition | Clear / Cloudy / Rain |
| Traffic Level | Low / Moderate |

### survey.csv (42 rows)
User perception survey covering: demographics, usage frequency, fare perception, peak hour awareness, frustrations, and comparison with Careem/Uber.

---

## Key Findings
1. **Negotiation Gap:** Final fares consistently exceed suggested fares by 11–15% across all ride types
2. **Night Dominance:** 38% of trips occur at night (6–10 PM) but fares are 60% higher
3. **Supply Imbalance:** Cars receive most driver bids (3.94 avg); rickshaws least (2.48 avg)
4. **User Frustration:** Long wait & driver rejections are tied as top complaints (survey)
5. **Competitive Edge:** 90% of users find inDrive same or cheaper than Careem/Uber

## Model Performance
- **Algorithm:** Random Forest Regressor
- **Features:** Suggested Fare, Driver Bids, Ride Type, Time, Weather, Traffic
- **R² Score:** ~0.97 (97% variance in final fare explained)
- **Key Insight:** Suggested Fare is the strongest predictor; time of day is second

---


*Developed by Rameesha Tariq · BS Artificial Intelligence (4th Semester) · Superior University Lahore*
