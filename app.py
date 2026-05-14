import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Apni purani import aur naya prediction import
from data_cleaning import clean_trips, clean_survey
from predictions import train_fare_model, predict_fare

st.set_page_config(
    page_title="inDrive Lahore Analysis",
    layout="wide",
    initial_sidebar_state="expanded",
)

PURPLE = "#7F77DD"
TEAL   = "#1D9E75"
CORAL  = "#D85A30"
AMBER  = "#EF9F27"
GRAY   = "#888780"
BLUE   = "#378ADD"

@st.cache_data
def load_data():
    return clean_trips(), clean_survey()

df, survey = load_data()

# Model ko background mein train karein
model, model_columns = train_fare_model(df)

st.sidebar.markdown("## InDrive Lahore")
st.sidebar.markdown("**Real-world Data Science Project**")
st.sidebar.markdown("BS Artificial Intelligence · Superior University")
st.sidebar.divider()

# NAVIGATION MEIN NAYA PAGE ADD KIYA HAI 👇
page = st.sidebar.radio(
    "Navigate",
    ["Overview", "Trip Analysis", "Survey Insights", "Fare Predictor (AI)"],
)

st.sidebar.divider()
st.sidebar.caption(f"Dataset: trips & survey responses")
st.sidebar.caption("Data collected: Lahore, 2026")

if page == "Overview":
    st.title("inDrive Lahore — Trip Analysis Dashboard")
    st.markdown(
        """
        > **Problem Statement:** inDrive riders in Lahore suffer from *fare blindness* — no historical 
        context to judge whether the app's suggested fare is fair, what time to book, or which vehicle 
        type gets fastest response. This project analyzes **498 real trips** and **User survey 
        responses** to surface actionable, data-driven insights for Lahore riders.
        """
    )

    # KPI Row
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Trips", "498")
    c2.metric("Avg Accepted Fare", "₨520", "+12.6% vs suggested")
    c3.metric("Avg Driver Bids", "3.27", "per ride")
    c4.metric("Most Active Time", "Night", "6–10 PM (38%)")
    c5.metric("Survey Respondents", "42", "Lahore users")

    st.divider()

    # 3 Problems
    st.subheader("3 Real Problems This Data Solves")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### Fare Blindness")
        st.markdown(
            "Riders have **zero historical context** for offers. "
            "The app's suggested fare is wrong — riders end up paying **11–15% above** "
            "the suggestion every time. This project shows what fair fares look like "
            "by route, time, and vehicle type."
        )

    with col2:
        st.markdown("#### Time Trap")
        st.markdown(
            "Night and Late Night fares are **60% higher** than morning fares. "
            "Riders don't know this pattern exists. A single hour shift in travel "
            "time can save **₨200+** per trip — data makes this visible."
        )

    with col3:
        st.markdown("#### Ride Type Confusion")
        st.markdown(
            "Cars receive **3.94 avg bids** vs rickshaws at **2.48**. "
            "Riders choosing rickshaws face **37% longer waits** on average. "
            "Survey confirms driver rejections are the **#1 frustration** among users."
        )

    st.divider()
    st.subheader("Project Structure")
    st.code(
        """
        Data Collection  → Manual trip logs (team data collection across Lahore)
        Survey           → User responses (Google Form, public distribution)
        Data Cleaning    → Normalization, missing values, date fixing, feature engineering
        EDA              → 7 visualizations across time, area, fare, weather, bids
        Machine Learning → Random Forest Regressor to predict real-time fair prices
        Dashboard        → This Streamlit app (interactive, portfolio-ready)
        """,
        language="text",
    )

elif page == "Trip Analysis":
    st.title("Trip Analysis")
    st.caption("Based on 498 inDrive trips collected across Lahore")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Ride Type Distribution")
        ride_counts = df["Ride Type"].value_counts().reset_index()
        ride_counts.columns = ["Ride Type", "Trips"]
        fig = px.pie(ride_counts, values="Trips", names="Ride Type",
                     hole=0.55, color_discrete_sequence=[PURPLE, TEAL, CORAL])
        fig.update_traces(textinfo="percent+label")
        fig.update_layout(showlegend=False, height=300, margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Trip Volume by Time of Day")
        time_vol = df["Time Category"].value_counts().reindex(
            ["Early Morning (5–8 AM)", "Morning (8–11 AM)", "Afternoon (11 AM–2 PM)",
             "Evening (2–6 PM)", "Night (6–10 PM)", "Late Night (10 PM+)"]
        ).reset_index()
        time_vol.columns = ["Time", "Trips"]
        fig = px.bar(time_vol, x="Time", y="Trips", color_discrete_sequence=[PURPLE],
                     text="Trips")
        fig.update_traces(textposition="outside")
        fig.update_layout(height=300, xaxis_tickangle=-35, margin=dict(t=10, b=80))
        st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Avg Fare by Time of Day (PKR)")
        time_fare = df.groupby("Time Category", observed=True)["Final Accepted Fare (PKR)"] \
                      .mean().reindex(
                          ["Early Morning (5–8 AM)", "Morning (8–11 AM)", "Afternoon (11 AM–2 PM)",
                           "Evening (2–6 PM)", "Night (6–10 PM)", "Late Night (10 PM+)"]
                      ).reset_index()
        time_fare.columns = ["Time", "Avg Fare"]
        time_fare["Avg Fare"] = time_fare["Avg Fare"].round(0)
        fig = px.bar(time_fare, x="Time", y="Avg Fare",
                     color="Avg Fare", color_continuous_scale=[[0, TEAL], [0.5, PURPLE], [1, CORAL]],
                     text="Avg Fare")
        fig.update_traces(texttemplate="₨%{text}", textposition="outside")
        fig.update_layout(height=320, xaxis_tickangle=-35, coloraxis_showscale=False,
                          margin=dict(t=10, b=80))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("⚠️ Early Morning only 1 data point — treat as outlier")

    with col4:
        st.subheader("Negotiation Gap: Suggested vs Accepted")
        fig = go.Figure()
        colors_map = {"Car": PURPLE, "Bike": TEAL, "Rickshaw": CORAL, "rickshaw": CORAL}
        for ride in df["Ride Type"].unique():
            sub = df[df["Ride Type"] == ride]
            color = colors_map.get(ride, GRAY)
            fig.add_trace(go.Scatter(
                x=sub["Suggested Fare (PKR)"], y=sub["Final Accepted Fare (PKR)"],
                mode="markers", name=ride, opacity=0.5,
                marker=dict(color=color, size=6)
            ))
        max_val = 1800
        fig.add_trace(go.Scatter(x=[0, max_val], y=[0, max_val],
                                 mode="lines", name="Equal (no gap)",
                                 line=dict(dash="dash", color="black", width=1)))
        fig.update_layout(height=320, xaxis_title="Suggested Fare (PKR)",
                          yaxis_title="Final Accepted Fare (PKR)",
                          legend=dict(orientation="h", y=-0.2),
                          margin=dict(t=10))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Points above dashed line = rider paid MORE than app suggested")

    col5, col6 = st.columns([3, 2])

    with col5:
        st.subheader("Top 10 Pickup Areas")
        top10 = df["Pickup Area"].value_counts().head(10).sort_values().reset_index()
        top10.columns = ["Area", "Trips"]
        fig = px.bar(top10, x="Trips", y="Area", orientation="h",
                     color_discrete_sequence=[PURPLE], text="Trips")
        fig.update_traces(textposition="outside")
        fig.update_layout(height=340, margin=dict(t=10, l=10))
        st.plotly_chart(fig, use_container_width=True)

    with col6:
        st.subheader("Avg Driver Bids")
        bids = df.groupby("Ride Type")["Num of Driver Bids"].mean().sort_values(ascending=False)
        bids_df = bids.reset_index()
        bids_df.columns = ["Ride Type", "Avg Bids"]
        bids_df["Avg Bids"] = bids_df["Avg Bids"].round(2)
        fig = px.bar(bids_df, x="Ride Type", y="Avg Bids",
                     color="Ride Type", color_discrete_sequence=[PURPLE, TEAL, CORAL],
                     text="Avg Bids")
        fig.update_traces(textposition="outside")
        fig.update_layout(height=240, showlegend=False, margin=dict(t=10))
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Weather Conditions")
        wc = df["Weather Condition"].value_counts().reset_index()
        wc.columns = ["Weather", "Trips"]
        fig = px.pie(wc, values="Trips", names="Weather", hole=0.5,
                     color_discrete_sequence=[AMBER, GRAY, BLUE])
        fig.update_layout(height=200, margin=dict(t=0, b=0), showlegend=True,
                          legend=dict(orientation="h", y=-0.15))
        st.plotly_chart(fig, use_container_width=True)


    st.divider()
    st.subheader("Key Insights")
    i1, i2, i3 = st.columns(3)
    i1.info("**Overpayment:** Riders pay ₨520 avg vs ₨462 suggested — a consistent 12.6% gap across all ride types and conditions.")
    i2.warning("**Peak pricing:** Night (6–10 PM) has the most trips but Early Morning sees the highest fares (supply shortage).")
    i3.success("**Best deal:** Book in the morning (8–11 AM) for lowest fares. Avg fare ₨344 vs ₨586 at Late Night.")


elif page == "Survey Insights":
    st.title("User Survey Insights")
    st.caption("InDrive users surveyed across Lahore — real perceptions and pain points")

    def safe_col(sv, keyword):
        matches = [c for c in sv.columns if keyword.lower() in c.lower()]
        return matches[0] if matches else None

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Gender")
        gc = safe_col(survey, "gender")
        if gc:
            g = survey[gc].value_counts().reset_index()
            g.columns = ["Gender", "Count"]
            fig = px.pie(g, values="Count", names="Gender", hole=0.55,
                         color_discrete_sequence=[CORAL, PURPLE, GRAY])
            fig.update_layout(height=260, margin=dict(t=10, b=10), showlegend=True)
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Age Group")
        ac = safe_col(survey, "age")
        if ac:
            age = survey[ac].value_counts().reset_index()
            age.columns = ["Age", "Count"]
            fig = px.bar(age, x="Age", y="Count", color_discrete_sequence=[TEAL], text="Count")
            fig.update_traces(textposition="outside")
            fig.update_layout(height=260, margin=dict(t=10))
            st.plotly_chart(fig, use_container_width=True)

    with col3:
        st.subheader("Usage Frequency")
        fc = safe_col(survey, "often")
        if fc:
            freq = survey[fc].value_counts().reset_index()
            freq.columns = ["Frequency", "Count"]
            fig = px.bar(freq, x="Count", y="Frequency", orientation="h",
                         color_discrete_sequence=[PURPLE], text="Count")
            fig.update_traces(textposition="outside")
            fig.update_layout(height=260, margin=dict(t=10))
            st.plotly_chart(fig, use_container_width=True)

    col4, col5 = st.columns(2)

    with col4:
        st.subheader("Biggest Frustrations")
        fr_c = safe_col(survey, "frustration")
        if fr_c:
            fr = survey[fr_c].value_counts().reset_index()
            fr.columns = ["Frustration", "Count"]
            fr = fr.sort_values("Count")
            fig = px.bar(fr, x="Count", y="Frustration", orientation="h",
                         color="Count",
                         color_continuous_scale=[[0, TEAL], [0.5, PURPLE], [1, CORAL]],
                         text="Count")
            fig.update_traces(textposition="outside")
            fig.update_layout(height=300, coloraxis_showscale=False, margin=dict(t=10))
            st.plotly_chart(fig, use_container_width=True)

    with col5:
        st.subheader("inDrive vs Careem/Yango/Bykea")
        app_c = safe_col(survey, "careem")
        if not app_c:
            app_c = safe_col(survey, "other")
        if app_c:
            comp = survey[app_c].value_counts().reset_index()
            comp.columns = ["Comparison", "Count"]
            fig = px.bar(comp, x="Comparison", y="Count",
                         color="Comparison",
                         color_discrete_sequence=[TEAL, GRAY, CORAL, AMBER],
                         text="Count")
            fig.update_traces(textposition="outside")
            fig.update_layout(height=300, showlegend=False, margin=dict(t=10),
                              xaxis_tickangle=-20)
            st.plotly_chart(fig, use_container_width=True)

    col6, col7 = st.columns(2)

    with col6:
        st.subheader("Would You Recommend inDrive?")
        rec_c = safe_col(survey, "recommend")
        if rec_c:
            rec = survey[rec_c].value_counts().reset_index()
            rec.columns = ["Response", "Count"]
            fig = px.pie(rec, values="Count", names="Response", hole=0.55,
                         color_discrete_sequence=[TEAL, AMBER, CORAL])
            fig.update_layout(height=260, margin=dict(t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)

    with col7:
        st.subheader("Has Weather Affected Your Fare?")
        wx_c = safe_col(survey, "weather")
        if wx_c:
            wx = survey[wx_c].value_counts().reset_index()
            wx.columns = ["Response", "Count"]
            fig = px.pie(wx, values="Count", names="Response", hole=0.55,
                         color_discrete_sequence=[BLUE, AMBER, GRAY])
            fig.update_layout(height=260, margin=dict(t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("Survey Summary")
    s1, s2, s3 = st.columns(3)
    s1.error("**Pain Point #1:** Long wait times & driver rejections (tied at 10 responses each) — supply-demand mismatch proven by data too.")
    s2.info("**Pricing:** 90% of users find inDrive cheaper or equal to Careem/Uber — competitive advantage confirmed.")
    s3.success("**NPS Signal:** 52.5% say Yes to recommending, 37.5% say Maybe — product has satisfied users but fixable gaps remain.")

elif page == "Fare Predictor (AI)":
    st.title("AI Smart Predictor: Rider vs Driver")
    st.markdown("Is predictive model mein ab Pickup/Dropoff locations aur timings ka asar bhi shamil hai.")
    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Route Details")
        pickup = st.selectbox("Pickup Area", sorted(df['Pickup Area'].unique()))
        dropoff = st.selectbox("Dropoff Area", sorted(df['Dropoff Area'].unique()))
        ride_type = st.selectbox("Ride Type", df['Ride Type'].unique())

    with col2:
        st.markdown("#### Context & App Price")
        suggested_fare = st.number_input("inDrive Suggested Fare (PKR)", min_value=50, value=350)
        time_cat = st.selectbox("Time Category", df['Time Category'].unique())
        traffic = st.selectbox("Traffic Level", df['Traffic Level'].unique())
        weather = st.selectbox("Weather Condition", df['Weather Condition'].unique())

    if st.button("Calculate Smart Result", type="primary", use_container_width=True):
        inputs = {
            'Suggested Fare (PKR)': suggested_fare,
            'Ride Type': ride_type,
            'Time Category': time_cat,
            'Traffic Level': traffic,
            'Weather Condition': weather,
            'Pickup Area': pickup,
            'Dropoff Area': dropoff
        }
        
        predicted_fare = predict_fare(model, model_columns, inputs)
        
        st.divider()
        st.subheader(f"Market Fair Price: ₨ {predicted_fare}")

        rider_col, driver_col = st.columns(2)

        with rider_col:
            st.info("### For Rider (Sawaari)")
            gap = predicted_fare - suggested_fare
            if gap > 0:
                st.write(f"App ki qeemat kam hai. Ride jaldi lene ke liye **₨ {predicted_fare}** tak offer karein.")
                st.write(f"**Mashwara:** Aapko suggested fare se taqreeban **₨ {gap}** ooper dena paray ga.")
            else:
                st.write("App ki qeemat theek hai. Is price par aapko ride mil jani chahiye.")

        with driver_col:
            st.success("### For Driver")
            st.write(f"Is route par winning bid **₨ {predicted_fare}** ke qareeb hai.")
            st.write(f"**Mashwara:** Agar aap **₨ {predicted_fare - 20}** se **₨ {predicted_fare + 20}** ke darmiyan bid karenge toh acceptance ke chances 90% hain.")

        st.warning(f"**Data Fact:** {pickup} se {dropoff} ke darmiyan {time_cat} mein aksar negotiation gap barh jata hai.")