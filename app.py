import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title("🌫️ Phân tích ô nhiễm không khí (PM2.5) - Delhi, Hyderabad, Amaravati")

# Load dữ liệu
@st.cache_data
def load_data():
    df = pd.read_csv("filtered_city_data.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    return df

data = load_data()

# Sidebar: chọn thành phố và thời gian
st.sidebar.header("Bộ lọc")
cities = st.sidebar.multiselect("Chọn thành phố", options=data["City"].unique(), default=data["City"].unique())
date_range = st.sidebar.date_input("Chọn khoảng thời gian", [data["Date"].min(), data["Date"].max()])

# Lọc dữ liệu
filtered_data = data[data["City"].isin(cities)]
filtered_data = filtered_data[(filtered_data["Date"] >= pd.to_datetime(date_range[0])) & 
                              (filtered_data["Date"] <= pd.to_datetime(date_range[1]))]

# Biểu đồ PM2.5 theo thời gian
st.subheader("📈 PM2.5 theo thời gian")
fig, ax = plt.subplots(figsize=(12, 5))
for city in cities:
    city_data = filtered_data[filtered_data["City"] == city]
    ax.plot(city_data["Date"], city_data["PM2.5"], label=city)
ax.set_xlabel("Thời gian")
ax.set_ylabel("PM2.5")
ax.legend()
ax.grid(True)
st.pyplot(fig)

# Trung bình theo tháng
st.subheader("📊 Trung bình PM2.5 theo tháng")
filtered_data["Month"] = filtered_data["Date"].dt.to_period("M")
monthly_avg = filtered_data.groupby(["Month", "City"])["PM2.5"].mean().reset_index()
monthly_avg["Month"] = monthly_avg["Month"].astype(str)

fig2, ax2 = plt.subplots(figsize=(12, 5))
for city in cities:
    subset = monthly_avg[monthly_avg["City"] == city]
    ax2.plot(subset["Month"], subset["PM2.5"], marker='o', label=city)
ax2.set_xticks(subset["Month"][::2])
ax2.set_xticklabels(subset["Month"][::2], rotation=45)
ax2.set_ylabel("PM2.5")
ax2.legend()
ax2.grid(True)
st.pyplot(fig2)

# So sánh phân bố
st.subheader("📦 So sánh phân bố PM2.5 giữa các thành phố")
st.box_plot_data = filtered_data[["City", "PM2.5"]]
st.box_plot_data = st.box_plot_data.dropna()
st.dataframe(st.box_plot_data.groupby("City")["PM2.5"].describe())

# Nút tải dữ liệu
st.download_button("📥 Tải dữ liệu đã lọc", data=filtered_data.to_csv(index=False), file_name="filtered_pm25.csv")
