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
        subset = monthly_avg[monthly_avg["City"] == city].reset_index(drop=True)
        ax2.plot(subset["Month"], subset[pollutant], marker='o', label=city)
    ax2.set_xticks(range(0, len(subset), 2))
    ax2.set_xticklabels(subset["Month"][::2], rotation=45)
    ax2.set_ylabel(pollutant)
    ax2.legend()
    ax2.grid(True)
    st.pyplot(fig2)

    st.subheader(f"📦 So sánh phân bố '{pollutant}' giữa các thành phố")
    st.dataframe(filtered.groupby("City")[pollutant].describe().round(2))

    st.download_button("📅 Tải dữ liệu đã lọc", data=filtered.to_csv(index=False), file_name="filtered_air_pollution_data.csv")

# ======================== TAB 2: DỰ ĐOÁN ========================
with tabs[1]:
    st.subheader("🔮 Dự đoán nồng độ PM2.5")

    pred_date = st.date_input("📅 Chọn ngày dự đoán", value=date(2020, 1, 1), 
                              min_value=data["Date"].min().date(), 
                              max_value=data["Date"].max().date())
    city = st.selectbox("🏩 Chọn thành phố", data["City"].unique())

    all_features = model.feature_names_in_ if hasattr(model, 'feature_names_in_') else [
        "City", "Day", "Month", "PM10", "NO", "NO2", "NOx", "NH3", "CO", "SO2", "O3", "Benzene"
    ]

    city_mapping = {c: idx for idx, c in enumerate(data["City"].unique())}

    default_inputs = {}
    for feature in all_features:
        if feature == "City":
            default_inputs[feature] = city_mapping.get(city, 0)
        elif feature == "Day":
            default_inputs[feature] = pred_date.day
        elif feature == "Month":
            default_inputs[feature] = pred_date.month
        elif feature in data.columns:
            try:
                default_inputs[feature] = round(pd.to_numeric(data[feature], errors='coerce').mean(), 2)
            except:
                default_inputs[feature] = 0
        else:
            default_inputs[feature] = 0

    for feature in all_features:
        if feature not in ["City", "Day", "Month"]:
            default_inputs[feature] = st.number_input(f"🔸 {feature}", value=default_inputs[feature])

    input_df = pd.DataFrame([{f: default_inputs[f] for f in all_features}])

    if st.button("🧮 Dự đoán PM2.5"):
        try:
            result = model.predict(input_df)
            st.success(f"✅ Dự đoán PM2.5: **{round(float(result[0]), 2)} µg/m³**")
        except Exception as e:
            st.error(f"❌ Lỗi khi dự đoán: {e}")
