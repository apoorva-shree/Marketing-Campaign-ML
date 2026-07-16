# -*- coding: utf-8 -*-
"""
Created on Thu Jul 16 12:01:43 2026

@author: HP
"""


import streamlit as st
import pandas as pd
import pickle


# ---------------------------
# Load Model & Encoder
# ---------------------------

import xgboost as xgb
import json

MODEL_PATH = r"C:\Users\HP\OneDrive\Documents\Marketing campaign analysis\Model\model.json"
FEATURES_PATH = r"C:\Users\HP\OneDrive\Documents\Marketing campaign analysis\Model\feature_names.json"
ENCODER_PATH = r"C:\Users\HP\OneDrive\Documents\Marketing campaign analysis\Model\encoder.pkl"

model = xgb.XGBClassifier()
model.load_model(MODEL_PATH)

with open(FEATURES_PATH, "r") as f:
    feature_names = json.load(f)

with open(ENCODER_PATH, "rb") as file:
    encoder = pickle.load(file)

# ---------------------------
# Page Configuration
# ---------------------------

st.set_page_config(
    page_title="Marketing Campaign Conversion Prediction",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Marketing Campaign Conversion Prediction")
st.write(
    "Enter the campaign details below to predict whether the customer is likely to convert."
)

st.divider()

# ---------------------------
# Input Section
# ---------------------------

col1, col2 = st.columns(2)

with col1:

    Age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=None,
        placeholder="Enter Age"
    )

    Gender = st.selectbox(
        "Gender",
        ["Male", "Female"],
        index=None,
        placeholder="Select Gender"
    )

    Income = st.number_input(
        "Income",
        min_value=0.0,
        value=None,
        placeholder="Enter Annual Income"
    )

    CampaignChannel = st.selectbox(
        "Campaign Channel",
        [
            "Email",
            "Social Media",
            "SEO",
            "PPC",
            "Referral"
        ],
        index=None,
        placeholder="Select Campaign Channel"
    )

    CampaignType = st.selectbox(
        "Campaign Type",
        [
            "Awareness",
            "Consideration",
            "Conversion",
            "Retention"
        ],
        index=None,
        placeholder="Select Campaign Type"
    )

    AdSpend = st.number_input(
        "Ad Spend",
        min_value=0.0,
        value=None,
        placeholder="Enter Ad Spend"
    )

    ClickThroughRate = st.number_input(
        "Click Through Rate",
        min_value=0.0,
        value=None,
        placeholder="Enter CTR"
    )

    ConversionRate = st.number_input(
        "Conversion Rate",
        min_value=0.0,
        value=None,
        placeholder="Enter Conversion Rate"
    )

with col2:

    WebsiteVisits = st.number_input(
        "Website Visits",
        min_value=0,
        value=None,
        placeholder="Enter Website Visits"
    )

    PagesPerVisit = st.number_input(
        "Pages Per Visit",
        min_value=0.0,
        value=None,
        placeholder="Enter Pages Per Visit"
    )

    TimeOnSite = st.number_input(
        "Time On Site",
        min_value=0.0,
        value=None,
        placeholder="Enter Time On Site"
    )

    SocialShares = st.number_input(
        "Social Shares",
        min_value=0,
        value=None,
        placeholder="Enter Social Shares"
    )

    EmailOpens = st.number_input(
        "Email Opens",
        min_value=0,
        value=None,
        placeholder="Enter Email Opens"
    )

    EmailClicks = st.number_input(
        "Email Clicks",
        min_value=0,
        value=None,
        placeholder="Enter Email Clicks"
    )

    PreviousPurchases = st.number_input(
        "Previous Purchases",
        min_value=0,
        value=None,
        placeholder="Enter Previous Purchases"
    )

    LoyaltyPoints = st.number_input(
        "Loyalty Points",
        min_value=0,
        value=None,
        placeholder="Enter Loyalty Points"
    )

st.divider()

# ---------------------------
# Prediction Button
# ---------------------------

if st.button("Predict Conversion"):

    # Check categorical inputs
    if Gender is None or CampaignChannel is None or CampaignType is None:
        st.warning("Please fill all the fields.")
        st.stop()

    # ---------- Feature Engineering ----------

    # Age Group
    if Age <= 25:
        Age_Group = "18-25"
    elif Age <= 35:
        Age_Group = "26-35"
    elif Age <= 45:
        Age_Group = "36-45"
    elif Age <= 60:
        Age_Group = "46-60"
    else:
        Age_Group = "60+"

    EngagementScore = (
        WebsiteVisits
        + PagesPerVisit
        + TimeOnSite
        + SocialShares
        + EmailOpens
        + EmailClicks
    )

    Estimated_Clicks = WebsiteVisits * ClickThroughRate

    Estimated_CPC = (
        AdSpend / Estimated_Clicks
        if Estimated_Clicks != 0 else 0
    )

    EmailEngagementRate = (
        EmailClicks / EmailOpens
        if EmailOpens != 0 else 0
    )

    # ---------- Input Data ----------

    input_df = pd.DataFrame({
        "Age":[Age],
        "Gender":[Gender],
        "Income":[Income],
        "CampaignChannel":[CampaignChannel],
        "CampaignType":[CampaignType],
        "AdSpend":[AdSpend],
        "ClickThroughRate":[ClickThroughRate],
        "ConversionRate":[ConversionRate],
        "WebsiteVisits":[WebsiteVisits],
        "PagesPerVisit":[PagesPerVisit],
        "TimeOnSite":[TimeOnSite],
        "SocialShares":[SocialShares],
        "EmailOpens":[EmailOpens],
        "EmailClicks":[EmailClicks],
        "PreviousPurchases":[PreviousPurchases],
        "LoyaltyPoints":[LoyaltyPoints],
        "EngagementScore":[EngagementScore],
        "Estimated_Clicks":[Estimated_Clicks],
        "Estimated_CPC":[Estimated_CPC],
        "EmailEngagementRate":[EmailEngagementRate],
        "Age_Group":[Age_Group]
    })

    # ---------- Encode ----------

    categorical_cols = [
        "Gender",
        "CampaignChannel",
        "CampaignType",
        "Age_Group"
    ]

    encoded = encoder.transform(input_df[categorical_cols])

    encoded_df = pd.DataFrame(
        encoded,
        columns=encoder.get_feature_names_out(categorical_cols)
    )

    numeric_df = input_df.drop(columns=categorical_cols).reset_index(drop=True)

    final_df = pd.concat([numeric_df, encoded_df], axis=1)

    # ---------- Arrange Columns ----------

    final_df = final_df.reindex(columns=feature_names, fill_value=0)

    # ---------- Prediction ----------

    prediction = model.predict(final_df)[0]
    probability = model.predict_proba(final_df)[0][1]

    if prediction == 1:
        st.success("✅ Customer is likely to convert.")
    else:
        st.error("❌ Customer is unlikely to convert.")

    st.metric("Conversion Probability", f"{probability:.2%}")
    st.divider()
    st.subheader("📊 Engineered KPIs")

    kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)

    with kpi_col1:
        st.metric("Engagement Score", f"{EngagementScore:.1f}")

    with kpi_col2:
        st.metric("Estimated Clicks", f"{Estimated_Clicks:.1f}")

    with kpi_col3:
        st.metric(
            "Estimated CPC",
            f"₹{Estimated_CPC:.2f}" if Estimated_Clicks != 0 else "N/A"
        )

    with kpi_col4:
        st.metric(
            "Email Engagement Rate",
            f"{EmailEngagementRate:.2%}" if EmailOpens != 0 else "N/A"
        )

    st.caption(f"Age Group: **{Age_Group}**")