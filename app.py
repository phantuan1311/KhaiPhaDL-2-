import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from datetime import date

st.set_page_config(layout="wide")
st.title("🌫️ Phân tích & Dự đoán chất lượng không khí (Ấn Độ - 2018 đến 2020)")

# Load dữ liệu
@st.cache_data
def load_data():
    df = pd.read_csv("filtered_city_data.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    return df

data = load_data()

# Load mô hình dự đoán PM2.5
@st.cache_resource
def load_model():
    return joblib.load("model_pm25.pkl")

model = load_model()

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

# ===================== DỰ ĐOÁN PM2.5 =====================
st.header("🔮 Dự đoán nồng độ bụi mịn PM2.5")
tab1, tab2 = st.tabs(["📉 Dự đoán nhanh (PM10 & NO2)", "🧪 Dự đoán chi tiết (toàn bộ đặc trưng)"])

with tab1:
    st.subheader("📉 Dự đoán từ thành phố + ngày + PM10 + NO2")
    col1, col2 = st.columns(2)
    with col1:
        pred_city = st.selectbox("Thành phố", data["City"].unique())
        pred_date = st.date_input("Ngày cần dự đoán", value=date(2020, 1, 1), min_value=data["Date"].min().date(), max_value=data["Date"].max().date())
    with col2:
        pm10 = st.number_input("PM10", value=100.0)
        no2 = st.number_input("NO2", value=40.0)

    pred_df_1 = pd.DataFrame({
        "City": [pred_city],
        "Day": [pred_date.day],
        "Month": [pred_date.month],
        "PM10": [pm10],
        "NO2": [no2]
    })
    if "City" in model.feature_names_in_:
        pred_df_1["City"] = pred_df_1["City"].astype("category").cat.codes

    if st.button("🧮 Dự đoán PM2.5 (từ PM10 & NO2)"):
        try:
            result = model.predict(pred_df_1)
            st.success(f"✅ PM2.5 ước tính: **{round(result[0], 2)} µg/m³**")
        except Exception as e:
            st.error(f"❌ Lỗi khi dự đoán: {e}")
            st.write("Columns:", pred_df_1.columns.tolist())

with tab2:
    st.subheader("🧪 Nhập toàn bộ thông số để dự đoán")
    col1, col2, col3 = st.columns(3)
    with col1:
        PM10 = st.number_input("PM10", value=50.0)
        NO2 = st.number_input("NO2", value=30.0)
        NO = st.number_input("NO", value=20.0)
        NOx = st.number_input("NOx", value=40.0)
    with col2:
        NH3 = st.number_input("NH3", value=10.0)
        CO = st.number_input("CO", value=0.5)
        SO2 = st.number_input("SO2", value=15.0)
        O3 = st.number_input("O3", value=25.0)
    with col3:
        Benzene = st.number_input("Benzene", value=5.0)
        Toluene = st.number_input("Toluene", value=5.0)
        Xylene = st.number_input("Xylene", value=5.0)
        Month = st.slider("Tháng", 1, 12, 6)

    pred_df_2 = pd.DataFrame([{
        "PM10": PM10, "NO2": NO2, "NO": NO, "NOx": NOx,
        "NH3": NH3, "CO": CO, "SO2": SO2, "O3": O3,
        "Benzene": Benzene, "Toluene": Toluene, "Xylene": Xylene,
        "Month": Month
    }])

    if st.button("🧬 Dự đoán PM2.5 (chi tiết)"):
        try:
            result = model.predict(pred_df_2)
            st.success(f"✅ PM2.5 ước tính: **{round(result[0], 2)} µg/m³**")
        except Exception as e:
            st.error(f"❌ Lỗi khi dự đoán: {e}")
            st.write("Columns:", pred_df_2.columns.tolist())
