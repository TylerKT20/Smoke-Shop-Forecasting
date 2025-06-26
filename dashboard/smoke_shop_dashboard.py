import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv("data/smoke_shop_transactions.csv", parse_dates=["date"])

# Title
st.title("Smoke Shop Sales Dashboard")

# Sidebar filters
st.sidebar.header("Filter Options")
selected_category = st.sidebar.selectbox("Select Category", options=["All"] + list(df["category"].unique()))
selected_product = st.sidebar.selectbox("Select Product", options=["All"] + list(df["product_name"].unique()))

# Filter data
filtered_df = df.copy()
if selected_category != "All":
    filtered_df = filtered_df[filtered_df["category"] == selected_category]
if selected_product != "All":
    filtered_df = filtered_df[filtered_df["product_name"] == selected_product]

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
fig, ax = plt.subplots()
weekly_sales.plot(ax=ax, figsize=(10, 4))
plt.ylabel("Revenue ($)")
plt.xlabel("Week")
st.pyplot(fig)

# Top-selling products
st.subheader("Top-Selling Products")
top_products = (
    filtered_df.groupby("product_name")["revenue"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

st.bar_chart(top_products)
