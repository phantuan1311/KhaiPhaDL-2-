import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib

# Load dữ liệu đã lọc
@st.cache_data
def load_data():
    return pd.read_csv("filtered_city_data.csv", parse_dates=["Date"])

df = load_data()

# Load mô hình dự đoán
model = joblib.load("model_pm25.pkl")

st.title("🌫️ Phân tích và Dự đoán Chất lượng Không khí")

# --- Lọc dữ liệu theo thành phố và khoảng thời gian ---
st.sidebar.header("🔍 Bộ lọc")

cities = df["City"].unique().tolist()
selected_city = st.sidebar.selectbox("Chọn thành phố", cities)

min_date = df["Date"].min()
max_date = df["Date"].max()
date_range = st.sidebar.date_input("Chọn khoảng thời gian", [min_date, max_date])

filtered = df[
    (df["City"] == selected_city) &
    (df["Date"] >= pd.to_datetime(date_range[0])) &
    (df["Date"] <= pd.to_datetime(date_range[1]))
].copy()

# --- Hiển thị dữ liệu ---
st.subheader(f"📊 Dữ liệu chất lượng không khí tại {selected_city}")
st.write(f"Số dòng dữ liệu: {len(filtered)}")
st.dataframe(filtered)

# --- Biểu đồ các biến ô nhiễm ---
pollutants = ["PM2.5", "PM10", "NO", "NO2", "NOx", "NH3", "CO", "SO2", "O3", "Benzene", "Toluene", "Xylene"]

selected_var = st.selectbox("Chọn biến ô nhiễm để xem biểu đồ", pollutants)
if selected_var in filtered.columns:
    fig, ax = plt.subplots()
    ax.plot(filtered["Date"], filtered[selected_var])
    ax.set_title(f"{selected_var} tại {selected_city}")
    ax.set_xlabel("Ngày")
    ax.set_ylabel(f"{selected_var} (µg/m³)")
    st.pyplot(fig)
else:
    st.warning("Biến này không tồn tại trong dữ liệu!")

# --- Dự đoán PM2.5 ---
st.subheader("🔮 Dự đoán PM2.5")

col1, col2 = st.columns(2)
with col1:
    pred_city = st.selectbox("Thành phố", cities, key="pred_city")
    pred_date = st.date_input("Ngày dự đoán", value=max_date, key="pred_date")
with col2:
    pm10 = st.number_input("Giá trị PM10", min_value=0.0, value=100.0)
    no2 = st.number_input("Giá trị NO2", min_value=0.0, value=30.0)

# Chuẩn bị input cho mô hình
pred_df = pd.DataFrame({
    "City": [pred_city],
    "Day": [pd.to_datetime(pred_date).day],
    "Month": [pd.to_datetime(pred_date).month],
    "PM10": [pm10],
    "NO2": [no2]
})
pred_df["City"] = pred_df["City"].astype("category").cat.codes
pred_df = pred_df[["City", "Day", "Month", "PM10", "NO2"]]

# --- Kiểm tra input và predict ---
if st.button("🧮 Dự đoán PM2.5"):
    st.write("📌 Kiểm tra input cho mô hình:")
    st.write("Số đặc trưng mô hình yêu cầu:", model.n_features_in_)
    st.write("Input shape:", pred_df.shape)
    st.write("Input columns:", pred_df.columns.tolist())

    try:
        result = model.predict(pred_df)
        st.success(f"✅ Dự đoán PM2.5: **{round(result[0], 2)} µg/m³**")
    except Exception as e:
        st.error(f"❌ Lỗi khi dự đoán: {e}")
