import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import streamlit as st

@st.cache_resource
def train_fare_model(df):
    features = ['Suggested Fare (PKR)', 'Ride Type', 'Time Category', 
                'Traffic Level', 'Weather Condition', 'Pickup Area', 'Dropoff Area']
    target = 'Final Accepted Fare (PKR)'
    
    model_df = df[features + [target]].dropna()
    
    X = model_df[features]
    y = model_df[target]

    X_encoded = pd.get_dummies(X, columns=['Ride Type', 'Time Category', 'Traffic Level', 
                                           'Weather Condition', 'Pickup Area', 'Dropoff Area'])
    
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_encoded, y)
    
    model_columns = X_encoded.columns
    return rf_model, model_columns

def predict_fare(model, model_columns, inputs):
    input_df = pd.DataFrame([inputs])

    input_encoded = pd.get_dummies(input_df, columns=['Ride Type', 'Time Category', 'Traffic Level', 
                                                      'Weather Condition', 'Pickup Area', 'Dropoff Area'])

    input_aligned = input_encoded.reindex(columns=model_columns, fill_value=0)
    
    prediction = model.predict(input_aligned)[0]
    return round(prediction)