import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from datetime import date

st.set_page_config(layout="wide")
st.title("🌫️ Phân tích & Dự đoán chất lượng không khí (Ấn Độ - 2018 đến 2020)")

# === Load dữ liệu ===
@st.cache_data
def load_data():
    df = pd.read_csv("filtered_city_data.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    return df

data = load_data()

# === Load model ===
@st.cache_resource
def load_model():
    return joblib.load("model_pm25.pkl")

model = load_model()

# === Giao diện ===
tabs = st.tabs(["📊 Phân tích dữ liệu", "🔮 Dự đoán PM2.5"])

# ======================== TAB 1: PHÂN TÍCH ========================
with tabs[0]:
    st.sidebar.header("🎛️ Bộ lọc dữ liệu")
    cities = st.sidebar.multiselect("Chọn thành phố", data["City"].unique(), default=data["City"].unique())
    date_range = st.sidebar.date_input("Khoảng thời gian", [data["Date"].min(), data["Date"].max()])

    numeric_cols = data.select_dtypes(include='number').columns.tolist()
    valid_cols = [col for col in numeric_cols if col not in ['AQI']]
    pollutant = st.sidebar.selectbox("Chọn biến cần phân tích", options=valid_cols)

    filtered = data[data["City"].isin(cities)]
    filtered = filtered[(filtered["Date"] >= pd.to_datetime(date_range[0])) & 
                        (filtered["Date"] <= pd.to_datetime(date_range[1]))]

    if filtered.empty:
        st.warning("⚠️ Không có dữ liệu trong khoảng bạn chọn.")
        st.stop()

    st.markdown("### 📌 Thống kê giá trị đầu/cuối khoảng thời gian")
    for city in cities:
        city_data = filtered[filtered["City"] == city].sort_values("Date")
        if not city_data.empty:
            col1, col2 = st.columns(2)
            with col1:
                st.metric(label=f"{city} - Đầu khoảng", value=round(city_data[pollutant].iloc[0], 2))
            with col2:
                st.metric(label=f"{city} - Cuối khoảng", value=round(city_data[pollutant].iloc[-1], 2))

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

    st.subheader(f"📦 So sánh phân bố '{pollutant}' giữa các thành phố")
    st.dataframe(filtered.groupby("City")[pollutant].describe().round(2))

    if "AQI_Bucket" in filtered.columns:
        st.subheader("🔍 Tần suất các mức AQI_Bucket")
        aqi_counts = filtered["AQI_Bucket"].value_counts()
        st.bar_chart(aqi_counts)

    st.download_button("📥 Tải dữ liệu đã lọc", data=filtered.to_csv(index=False), file_name="filtered_air_pollution_data.csv")

# ======================== TAB 2: DỰ ĐOÁN ========================
with tabs[1]:
    st.subheader("🔮 Dự đoán nồng độ PM2.5")

    pred_date = st.date_input("📅 Chọn ngày dự đoán", value=date(2020, 1, 1), 
                              min_value=data["Date"].min().date(), 
                              max_value=data["Date"].max().date())
    pm10 = st.number_input("Giá trị PM10", value=100.0)
    no2 = st.number_input("Giá trị NO2", value=40.0)

    city_mapping = {city: idx for idx, city in enumerate(data["City"].unique())}
    city_list = list(city_mapping.values())
    city_names = list(city_mapping.keys())

    pred_base_full = data.drop(columns=["Date", "PM2.5", "AQI", "AQI_Bucket"], errors="ignore").copy()
    feature_cols = pred_base_full.columns.tolist()
    default_values = pred_base_full.mean(numeric_only=True).to_dict()

    pred_base = []
    for city_id in city_list:
        row = default_values.copy()
        row.update({
            "City": city_id,
            "Day": pred_date.day,
            "Month": pred_date.month,
            "PM10": pm10,
            "NO2": no2
        })
        pred_base.append(row)

    pred_df = pd.DataFrame(pred_base)[feature_cols]

    if st.button("🧮 Dự đoán PM2.5 cho tất cả thành phố"):
        try:
            results = model.predict(pred_df)
            for city, val in zip(city_names, results):
                st.success(f"✅ {city}: {round(float(val), 2)} µg/m³")
        except Exception as e:
            st.error(f"❌ Lỗi khi dự đoán: {e}")

    st.markdown("---")
    st.subheader("🧪 Dự đoán theo tham số tùy chọn")
    st.info("🔧 Nhập PM10, NO2, và thời gian để dự đoán")

    custom_pm10 = st.number_input("PM10", value=pm10)
    custom_no2 = st.number_input("NO2", value=no2)
    custom_day = st.number_input("Day", value=pred_date.day, min_value=1, max_value=31)
    custom_month = st.number_input("Month", value=pred_date.month, min_value=1, max_value=12)
    custom_city = st.selectbox("Thành phố", options=city_names)

    if st.button("🧮 Dự đoán PM2.5 với tham số tùy chọn"):
        try:
            input_row = default_values.copy()
            input_row.update({
                "City": city_mapping[custom_city],
                "Day": custom_day,
                "Month": custom_month,
                "PM10": custom_pm10,
                "NO2": custom_no2
            })
            input_df = pd.DataFrame([input_row])[feature_cols]
            result = model.predict(input_df)
            st.success(f"✅ PM2.5 dự đoán: **{round(float(result[0]), 2)} µg/m³**")
        except Exception as e:
            st.error(f"❌ Lỗi khi dự đoán: {e}")
