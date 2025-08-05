import streamlit as st
import pandas as pd
import joblib
from datetime import datetime

# --------------------------
# Tải dữ liệu & mô hình
# --------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("filtered_city_data.csv")
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    return df

@st.cache_resource
def load_model():
    return joblib.load("pm25_model.pkl")  # Đổi tên đúng nếu cần

df = load_data()
model = load_model()

# --------------------------
# Sidebar – Thông tin
# --------------------------
st.sidebar.title("📊 Dự đoán PM2.5")
st.sidebar.write("Nhập các thông số ô nhiễm để dự đoán nồng độ bụi mịn PM2.5 (µg/m³)")

# --------------------------
# Nhập đầu vào
# --------------------------
with st.sidebar:
    st.subheader("🔧 Thông số đầu vào")
    pm10 = st.number_input("PM10", value=100.0)
    no2 = st.number_input("NO2", value=40.0)
    no = st.number_input("NO", value=25.0)
    nox = st.number_input("NOx", value=60.0)
    nh3 = st.number_input("NH3", value=10.0)
    co = st.number_input("CO", value=0.9)
    so2 = st.number_input("SO2", value=20.0)
    o3 = st.number_input("O3", value=30.0)
    benzene = st.number_input("Benzene", value=2.0)
    toluene = st.number_input("Toluene", value=10.0)
    xylene = st.number_input("Xylene", value=1.5)
    month = st.selectbox("Tháng", options=list(range(1, 13)), index=datetime.now().month - 1)

# --------------------------
# Dự đoán PM2.5
# --------------------------
st.header("📈 Dự đoán nồng độ PM2.5")

# Tạo DataFrame từ input
input_dict = {
    "PM10": pm10,
    "NO2": no2,
    "NO": no,
    "NOx": nox,
    "NH3": nh3,
    "CO": co,
    "SO2": so2,
    "O3": o3,
    "Benzene": benzene,
    "Toluene": toluene,
    "Xylene": xylene,
    "Month": month
}
pred_df = pd.DataFrame([input_dict])

# Ép kiểu số
for col in pred_df.columns:
    try:
        pred_df[col] = pd.to_numeric(pred_df[col])
    except:
        st.error(f"❌ Cột {col} không hợp lệ. Vui lòng nhập dạng số.")
        st.stop()

# Hiển thị input
st.subheader("🔍 Đầu vào cho mô hình:")
st.dataframe(pred_df)

# Kiểm tra kích thước input với mô hình
st.write("🧪 Số đặc trưng mô hình yêu cầu:", model.n_features_in_)
st.write("📐 Input shape:", pred_df.shape)
st.write("📋 Input columns:", pred_df.columns.tolist())

# Dự đoán
try:
    result = model.predict(pred_df)
    st.success(f"✅ Dự đoán PM2.5: {round(result[0], 2)} µg/m³")
except Exception as e:
    st.error(f"❌ Lỗi khi dự đoán: {e}")
