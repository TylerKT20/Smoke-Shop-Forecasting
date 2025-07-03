import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
from prophet.plot import plot_plotly
import plotly.graph_objs as go

# Load the data
df = pd.read_csv("data/smoke_shop_transactions.csv", parse_dates=["date"])

# Title
st.title("Smoke Shop Sales Dashboard")

# Sidebar filters
st.sidebar.header("Filter Options")
selected_category = st.sidebar.selectbox("Select Category", options=["All"] + list(df["category"].unique()))
selected_brand = st.sidebar.selectbox("Select Brand", options=["All"] + list(df["brand"].unique()))
selected_product = st.sidebar.selectbox("Select Product", options=["All"] + list(df["product_name"].unique()))
selected_item = st.sidebar.selectbox("Select Item", options=["All"] + list(df["item_name"].unique()))

# Filter data
filtered_df = df.copy()
if selected_category != "All":
    filtered_df = filtered_df[filtered_df["category"] == selected_category]
if selected_brand != "All":
    filtered_df = filtered_df[filtered_df["brand"] == selected_brand]
if selected_product != "All":
    filtered_df = filtered_df[filtered_df["product_name"] == selected_product]
if selected_item != "All":
    filtered_df = filtered_df[filtered_df["item_name"] == selected_item]

# Forecast Section
if selected_item != "All":
    st.subheader(f"Weekly Demand Forecast for {selected_item}")

    if st.checkbox("Show Demand Forecast", value=True):
        forecast_df = df[df["item_name"] == selected_item].copy()
        forecast_df["week"] = forecast_df["date"].dt.to_period("W").apply(lambda r: r.start_time)
        weekly_demand = forecast_df.groupby("week")["quantity"].sum().reset_index()

        if not weekly_demand.empty:
            weekly_demand.columns = ["week", "quantity"]
            prophet_df = weekly_demand.rename(columns={"week": "ds", "quantity": "y"})

            m = Prophet()
            m.fit(prophet_df)

            future = m.make_future_dataframe(periods=6, freq='W')
            forecast = m.predict(future)

            next_week = forecast.iloc[-1]
            next_month = forecast.iloc[-4:]

            st.markdown(f"""
            **Forecast Summary:**
            - Predicted demand for next week: **{int(next_week['yhat'])} units**
            - Average weekly demand for next month: **{int(next_month['yhat'].mean())} units**
            """)

            fig = plot_plotly(m, forecast)
            st.plotly_chart(fig, use_container_width=True)

            renamed_forecast = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].rename(columns={
                "ds": "week",
                "yhat": "forecast_quantity",
                "yhat_lower": "forecast_quantity_lower",
                "yhat_upper": "forecast_quantity_upper"
            })
            forecast_csv = renamed_forecast.to_csv(index=False).encode('utf-8')

            st.download_button(
                label="Download Forecast CSV",
                data=forecast_csv,
                file_name=f"{selected_item}_forecast.csv",
                mime='text/csv'
            )
        else:
            st.warning("Not enough data to forecast this item.")
else:
    # Optional: provide clearer instructions when item isn't selected
    if selected_product != "All":
        st.subheader(f"Weekly Demand Forecast for {selected_product}")
        st.info("Select a specific item to see forecast.")
    elif selected_brand != "All":
        st.subheader(f"Weekly Demand Forecast for {selected_brand}")
        st.info("Select a specific product and item to see forecast.")
    elif selected_category != "All":
        st.subheader(f"Weekly Demand Forecast for {selected_category}")
        st.info("Select a specific product and item to see forecast.")
    else:
        st.subheader("Weekly Demand Forecast")
        st.info("Select a specific item to see forecast.")

# KPIs
total_revenue = filtered_df["revenue"].sum()
total_quantity = filtered_df["quantity"].sum()
num_transactions = len(filtered_df)

st.metric("Total Revenue ($)", f"{total_revenue:,.2f}")
st.metric("Total Quantity Sold", total_quantity)
st.metric("Number of Transactions", num_transactions)

# Weekly revenue trend
df["week"] = df["date"].dt.to_period("W").apply(lambda r: r.start_time)
filtered_df["week"] = filtered_df["date"].dt.to_period("W").apply(lambda r: r.start_time)
weekly_sales = filtered_df.groupby("week")["revenue"].sum()

st.subheader("Weekly Revenue Trend")
if not weekly_sales.empty:
    fig, ax = plt.subplots()
    weekly_sales.plot(ax=ax, figsize=(10, 4))
    plt.ylabel("Revenue ($)")
    plt.xlabel("Week")
    st.pyplot(fig)
else:
    st.warning("No weekly revenue data available for this item.")

# Top-selling products
st.subheader("Top-Selling Products")
if selected_category != "All":
    top_df = df[df["category"] == selected_category]
else:
    top_df = df.copy()

top_items = (
    filtered_df.groupby("item_name")["revenue"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)
st.bar_chart(top_items)