import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import pickle

st.set_page_config(page_title="Real Estate Forecast Dashboard", layout="wide")
st.title("India Housing Analytics & Forecast Dashboard")

@st.cache_data
def load_data():
    return pd.read_csv("data/india_housing_prices.csv")

@st.cache_resource
def load_model(path):
    try:
        return pickle.load(open(path, "rb"))
    except:
        return None

df = load_data()
reg_model = load_model("model_regression.pkl")
clf_model = load_model("model_classifier.pkl")

def to_np(m):
    if isinstance(m, dict):
        for k in ("weights", "mean", "std"):
            if k in m:
                m[k] = np.array(m[k])
    return m

reg_model = to_np(reg_model)
clf_model = to_np(clf_model)

median_pps = df["Price_per_SqFt"].median()
default_pps = max(100, int(median_pps))

st.sidebar.header("Filters")
states = st.sidebar.multiselect("State", df["State"].unique())
cities = st.sidebar.multiselect("City", df["City"].unique())

df_f = df.copy()
if states:
    df_f = df_f[df_f["State"].isin(states)]
if cities:
    df_f = df_f[df_f["City"].isin(cities)]

st.subheader("Market Overview")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Listings", len(df_f))
c2.metric("Avg Price (Lakhs)", round(df_f["Price_in_Lakhs"].mean(), 2))
c3.metric("Avg Size (SqFt)", round(df_f["Size_in_SqFt"].mean(), 2))
c4.metric("Median PPS", round(df_f["Price_per_SqFt"].median(), 2))

st.subheader("1. Price Distribution")
fig1 = px.histogram(df_f, x="Price_in_Lakhs", nbins=50, color="City")
st.plotly_chart(fig1, use_container_width=True)

st.subheader("2. Market Position — Size vs PPS")
fig2 = px.scatter(df_f, x="Size_in_SqFt", y="Price_per_SqFt", size="Price_in_Lakhs", color="Property_Type", hover_data=["City", "BHK"])
st.plotly_chart(fig2, use_container_width=True)

st.subheader("3. Average Price by City")
city_avg = df_f.groupby("City")["Price_in_Lakhs"].mean().reset_index()
fig3 = px.bar(city_avg, x="City", y="Price_in_Lakhs", color="City")
st.plotly_chart(fig3, use_container_width=True)

st.subheader("4. Investment Heatmap (BHK vs City)")
df_f["Good_Investment"] = (df_f["Price_per_SqFt"] <= df_f["Price_per_SqFt"].median()).astype(int)
fig4 = px.density_heatmap(df_f, x="BHK", y="City", z="Good_Investment", color_continuous_scale="Viridis")
st.plotly_chart(fig4, use_container_width=True)

st.subheader("5. Price Trend by Build Year")
trend = df_f.groupby("Build_Year")["Price_in_Lakhs"].mean().reset_index()
fig5 = px.line(trend, x="Build_Year", y="Price_in_Lakhs", markers=True)
st.plotly_chart(fig5, use_container_width=True)

st.subheader("6. Property Type Distribution")
fig6 = px.pie(df_f, names="Property_Type", hole=0.4)
st.plotly_chart(fig6, use_container_width=True)

st.header("Predict Future Price & Investment Decision")

col1, col2 = st.columns(2)

with col1:
    bhk = st.number_input("BHK", 1, 10, 2)
    size = st.number_input("Size SqFt", 200, 10000, 800)
    pps = st.number_input("Price per SqFt", 100, 100000, default_pps)
    age = st.number_input("Age of Property", 0, 50, 5)
    state = st.selectbox("State", df["State"].unique())
    city = st.selectbox("City", df["City"].unique())
    ptype = st.selectbox("Property Type", df["Property_Type"].unique())
    furn = st.selectbox("Furnished", df["Furnished_Status"].unique())

with col2:
    st.write("### Future Price Settings")
    years = st.slider("Years ahead", 1, 10, 5)
    rate = st.slider("Growth Rate (%)", 1, 20, 8) / 100

current_price = (size * pps) / 100000

input_row = pd.DataFrame([{
    "BHK": bhk,
    "Size_in_SqFt": size,
    "Price_in_Lakhs": current_price,
    "Price_per_SqFt": pps,
    "Year_Built": 2025 - age,
    "Floor_No": 1,
    "Total_Floors": 10,
    "Age_of_Property": age,
    "Nearby_Schools": 5,
    "Nearby_Hospitals": 5,
    "Build_Year": 2025 - age,
    "Build_Decade": (2025 - age) // 10 * 10,
    "State": state,
    "City": city,
    "Property_Type": ptype,
    "Furnished_Status": furn
}])

if st.button("Predict Future Price"):
    future_price = current_price * ((1 + rate) ** years)

    st.subheader("Prediction Summary (KPIs)")
    k1, k2, k3 = st.columns(3)

    k1.metric("Current Estimated Price (Lakhs)", f"{round(current_price, 2)}")
    k2.metric(f"Future Price After {years} Years", f"{round(future_price, 2)}", delta=f"{round((future_price - current_price), 2)} Lakhs")
    appreciation = ((future_price - current_price) / current_price) * 100
    k3.metric("Growth (%)", f"{round(appreciation, 2)}%")

    st.subheader("Investment Recommendation")
    if pps <= median_pps:
        st.success("This appears to be a GOOD investment based on PPS and market averages.")
    else:
        st.warning("This may NOT be an ideal investment. Price per SqFt is above the market median.")
