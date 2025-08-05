import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title("🌫️ Phân tích chất lượng không khí (Ấn Độ - 2018 đến 2020)")

# Load dữ liệu
@st.cache_data
def load_data():
    df = pd.read_csv("filtered_city_data.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    return df

data = load_data()

# Sidebar: Chọn thành phố, thời gian, biến số
st.sidebar.header("🎛️ Bộ lọc dữ liệu")
cities = st.sidebar.multiselect("Chọn thành phố", data["City"].unique(), default=data["City"].unique())
date_range = st.sidebar.date_input("Khoảng thời gian", [data["Date"].min(), data["Date"].max()])

# Danh sách biến số loại trừ AQI_Bucket
numeric_cols = data.select_dtypes(include='number').columns.tolist()
valid_cols = [col for col in numeric_cols if col not in ['AQI']]  # giữ lại PM2.5, PM10, NO2,...
pollutant = st.sidebar.selectbox("Chọn biến cần phân tích", options=valid_cols)

# Lọc dữ liệu
filtered = data[data["City"].isin(cities)]
filtered = filtered[(filtered["Date"] >= pd.to_datetime(date_range[0])) & 
                    (filtered["Date"] <= pd.to_datetime(date_range[1]))]

# Kiểm tra dữ liệu rỗng
if filtered.empty:
    st.warning("⚠️ Không có dữ liệu trong khoảng bạn chọn.")
    st.stop()

# Thống kê đầu/cuối khoảng
st.markdown("### 📌 Thống kê giá trị đầu/cuối khoảng thời gian")
for city in cities:
    city_data = filtered[filtered["City"] == city].sort_values("Date")
    if not city_data.empty:
        col1, col2 = st.columns(2)
        with col1:
            st.metric(label=f"{city} - Đầu khoảng", value=round(city_data[pollutant].iloc[0], 2))
        with col2:
            st.metric(label=f"{city} - Cuối khoảng", value=round(city_data[pollutant].iloc[-1], 2))

# Biểu đồ theo thời gian
st.subheader(f"📈 Biến '{pollutant}' theo thời gian")
fig1, ax1 = plt.subplots(figsize=(12, 5))
for city in cities:
    city_data = filtered[filtered["City"] == city]
    ax1.plot(city_data["Date"], city_data[pollutant], label=city)
ax1.set_xlabel("Ngày")
ax1.set_ylabel(pollutant)
ax1.legend()
ax1.grid(True)
st.pyplot(fig1)

# Trung bình theo tháng
st.subheader(f"📊 Trung bình '{pollutant}' theo tháng")
filtered["Month"] = filtered["Date"].dt.to_period("M")
monthly_avg = filtered.groupby(["Month", "City"])[pollutant].mean().reset_index()
monthly_avg["Month"] = monthly_avg["Month"].astype(str)

fig2, ax2 = plt.subplots(figsize=(12, 5))
for city in cities:
    subset = monthly_avg[monthly_avg["City"] == city]
    ax2.plot(subset["Month"], subset[pollutant], marker='o', label=city)
ax2.set_xticks(subset["Month"][::2])
ax2.set_xticklabels(subset["Month"][::2], rotation=45)
ax2.set_ylabel(pollutant)
ax2.legend()
ax2.grid(True)
st.pyplot(fig2)

# So sánh phân bố
st.subheader(f"📦 So sánh phân bố '{pollutant}' giữa các thành phố")
st.dataframe(filtered.groupby("City")[pollutant].describe().round(2))

# Phân tích AQI_Bucket nếu tồn tại
if "AQI_Bucket" in filtered.columns:
    st.subheader("🔍 Tần suất các mức AQI_Bucket")
    aqi_counts = filtered["AQI_Bucket"].value_counts()
    st.bar_chart(aqi_counts)

# Nút tải dữ liệu
st.download_button("📥 Tải dữ liệu đã lọc", data=filtered.to_csv(index=False), file_name="filtered_air_pollution_data.csv")


import joblib
from datetime import date

# === DỰ ĐOÁN PM2.5 DỰA TRÊN MÔ HÌNH HUẤN LUYỆN ===
st.header("🔮 Dự đoán nồng độ PM2.5")

# Tải mô hình
@st.cache_resource
def load_model():
    return joblib.load("model_pm25.pkl")  # thay bằng tên file của bạn

model = load_model()

# Input
col1, col2 = st.columns(2)
with col1:
    pred_city = st.selectbox("Chọn thành phố", data["City"].unique())
    pred_date = st.date_input("Chọn ngày", value=date(2020, 1, 1), min_value=data["Date"].min().date(), max_value=data["Date"].max().date())

with col2:
    pm10 = st.number_input("Giá trị PM10", value=100.0)
    no2 = st.number_input("Giá trị NO2", value=40.0)

# Chuẩn bị input
pred_df = pd.DataFrame({
    "City": [pred_city],
    "Day": [pred_date.day],
    "Month": [pred_date.month],
    "PM10": [pm10],
    "NO2": [no2]
})

# Mã hoá giống khi training
pred_df["City"] = pred_df["City"].astype("category").cat.codes

# Đảm bảo đúng thứ tự cột như khi training
pred_df = pred_df[["City", "Day", "Month", "PM10", "NO2"]]

# Dự đoán
if st.button("🧮 Dự đoán PM2.5"):
    st.write("Input dùng cho mô hình:")
    st.dataframe(pred_df)
    result = model.predict(pred_df)
    st.success(f"✅ Dự đoán PM2.5: **{round(result[0], 2)} µg/m³**")




