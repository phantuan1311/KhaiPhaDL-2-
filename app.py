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

    pred_base = pd.DataFrame({
        "City": city_list,
        "Day": [pred_date.day] * len(city_list),
        "Month": [pred_date.month] * len(city_list),
        "PM10": [pm10] * len(city_list),
        "NO2": [no2] * len(city_list)
    })

    if st.button("🧮 Dự đoán PM2.5 cho tất cả thành phố"):
        try:
            results = model.predict(pred_base.values)
            for city, val in zip(city_names, results):
                st.success(f"✅ {city}: {round(float(val), 2)} µg/m³")
        except Exception as e:
            st.error(f"❌ Lỗi khi dự đoán: {e}")

    st.markdown("---")
    st.subheader("🧪 Dự đoán với đầy đủ đặc trưng (12 biến)")
    st.info("🔧 Nhập các thông số để mô hình dự đoán PM2.5")

    inputs = {}
    full_features = ["PM10", "NO2", "NO", "NOx", "NH3", "CO", "SO2", "O3", "Benzene", "Toluene", "Xylene", "Month"]
    for feat in full_features:
        default_val = pred_date.month if feat == "Month" else 0.0
        inputs[feat] = st.number_input(f"{feat}", value=float(default_val))

    pred_df_full = pd.DataFrame([inputs])

    if st.button("🧮 Dự đoán PM2.5 (đầy đủ đặc trưng)"):
        try:
            result = model.predict(pred_df_full.values)
            st.success(f"✅ Dự đoán PM2.5: **{round(float(result[0]), 2)} µg/m³**")
        except Exception as e:
            st.error(f"❌ Lỗi khi dự đoán: {e}")
