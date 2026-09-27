"""Analytics Dashboard page."""
from __future__ import annotations

import os
import pandas as pd
import streamlit as st

from ui.helpers import page_header

@st.cache_data
def load_data() -> pd.DataFrame | None:
    data_path = "data/raw/application_train.csv"
    if not os.path.exists(data_path):
        data_path = "data/synthetic/application_train_synthetic.csv"
        
    if not os.path.exists(data_path):
        return None
    return pd.read_csv(data_path)

def render() -> None:
    page_header(
        "Analytics Dashboard",
        "Business metrics and data distributions across the applicant pool."
    )
    
    df = load_data()
    if df is None:
        st.warning("No data found to display analytics. Please download the dataset first.")
        return
        
    st.subheader("Key Metrics")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Applicants", f"{len(df):,}")
    
    if "TARGET" in df.columns:
        positive_rate = df["TARGET"].mean() * 100
        col2.metric("Default Rate", f"{positive_rate:.2f}%")
        
    if "AMT_CREDIT" in df.columns:
        avg_credit = df["AMT_CREDIT"].mean()
        col3.metric("Average Credit Amount", f"${avg_credit:,.2f}")
        
    st.divider()
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.subheader("Income Type vs Default Rate (%)")
        if "NAME_INCOME_TYPE" in df.columns and "TARGET" in df.columns:
            income_target = df.groupby("NAME_INCOME_TYPE")["TARGET"].mean().sort_values(ascending=False) * 100
            st.bar_chart(income_target)

    with col_b:
        st.subheader("Applicant Age (Years) Distribution")
        if "DAYS_BIRTH" in df.columns:
            ages = (df["DAYS_BIRTH"] / -365.25).astype(int)
            st.bar_chart(ages.value_counts().sort_index())
