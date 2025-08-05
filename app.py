import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib

# Load dữ liệu đã lọc
@st.cache_data
def load_data():
    return pd.read_csv("filtered_city_data.csv", parse_dates=["Date"])

df = load_data()

# Load mô hình
model = joblib.load("model_pm25.pkl")

st.title("🌫️ Phân tích và Dự đoán PM2.5")

# --- Sidebar: Chọn thành phố và khoảng thời gian ---
st.sidebar.header("🔍 Bộ lọc dữ liệu")
cities = df["City"].unique().tolist()
selected_city = st.sidebar.selectbox("Chọn thành phố", cities)

min_date = df["Date"].min()
max_date = df["Date"].max()
date_range = st.sidebar.date_input("Khoảng thời gian", [min_date, max_date])

filtered = df[
    (df["City"] == selected_city) &
    (df["Date"] >= pd.to_datetime(date_range[0])) &
    (df["Date"] <= pd.to_datetime(date_range[1]))
].copy()

# --- Hiển thị dữ liệu ---
st.subheader(f"📊 Dữ liệu tại {selected_city}")
st.write(f"Số dòng dữ liệu: {len(filtered)}")
st.dataframe(filtered)

# --- Biểu đồ biến ô nhiễm ---
pollutants = ["PM2.5", "PM10", "NO", "NO2", "NOx", "NH3", "CO", "SO2", "O3", "Benzene", "Toluene", "Xylene"]
selected_var = st.selectbox("Chọn biến để hiển thị biểu đồ", pollutants)

if selected_var in filtered.columns:
    fig, ax = plt.subplots()
    ax.plot(filtered["Date"], filtered[selected_var])
    ax.set_title(f"{selected_var} tại {selected_city}")
    ax.set_xlabel("Ngày")
    ax.set_ylabel(f"{selected_var} (µg/m³)")
    st.pyplot(fig)
else:
    st.warning("Biến không tồn tại trong dữ liệu!")

# --- Dự đoán PM2.5 ---
st.subheader("🔮 Dự đoán PM2.5")

col1, col2 = st.columns(2)
with col1:
    pred_date = st.date_input("Ngày dự đoán", value=max_date)
with col2:
    pred_month = pd.to_datetime(pred_date).month

# --- Nhập các biến đặc trưng cần thiết cho mô hình ---
st.markdown("### 📥 Nhập các giá trị ô nhiễm:")
pm10 = st.number_input("PM10", min_value=0.0, value=100.0)
no2 = st.number_input("NO2", min_value=0.0, value=25.0)
no = st.number_input("NO", min_value=0.0, value=15.0)
nox = st.number_input("NOx", min_value=0.0, value=35.0)
nh3 = st.number_input("NH3", min_value=0.0, value=10.0)
co = st.number_input("CO", min_value=0.0, value=1.0)
so2 = st.number_input("SO2", min_value=0.0, value=5.0)
o3 = st.number_input("O3", min_value=0.0, value=30.0)
benzene = st.number_input("Benzene", min_value=0.0, value=2.0)
toluene = st.number_input("Toluene", min_value=0.0, value=10.0)
xylene = st.number_input("Xylene", min_value=0.0, value=1.0)

# --- Chuẩn bị dữ liệu dự đoán ---
pred_df = pd.DataFrame({
    "PM10": [pm10],
    "NO2": [no2],
    "NO": [no],
    "NOx": [nox],
    "NH3": [nh3],
    "CO": [co],
    "SO2": [so2],
    "O3": [o3],
    "Benzene": [benzene],
    "Toluene": [toluene],
    "Xylene": [xylene],
    "Month": [pred_month]
})

# --- Dự đoán ---
if st.button("🧮 Dự đoán PM2.5"):
    st.write("📌 Kiểm tra input:")
    st.write("Số đặc trưng mô hình yêu cầu:", model.n_features_in_)
    st.write("Input shape:", pred_df.shape)
    st.write("Input columns:", pred_df.columns.tolist())

    try:
        result = model.predict(pred_df)
        st.success(f"✅ Kết quả dự đoán PM2.5: **{round(result[0], 2)} µg/m³**")
    except Exception as e:
        st.error(f"❌ Lỗi khi dự đoán: {e}")
