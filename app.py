import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib

st.set_page_config(page_title="Phân tích và dự đoán PM2.5", layout="wide")

@st.cache_data
def load_data():
    return pd.read_csv("filtered_city_data.csv")

@st.cache_resource
def load_model():
    return joblib.load("model_pm25.pkl")

df = load_data()
model = load_model()

st.title("🌫️ Ứng dụng phân tích & dự đoán PM2.5")

page = st.sidebar.radio("📂 Chọn chức năng", ["📊 Khám phá dữ liệu", "🔮 Dự đoán PM2.5"])

if page == "📊 Khám phá dữ liệu":
    st.header("🔍 Khám phá dữ liệu ô nhiễm không khí")
    
    with st.expander("📅 Lọc dữ liệu theo thời gian"):
        min_date = pd.to_datetime(df["Date"]).min()
        max_date = pd.to_datetime(df["Date"]).max()
        start_date, end_date = st.date_input("Chọn khoảng thời gian", [min_date, max_date])
        df["Date"] = pd.to_datetime(df["Date"])
        df_filtered = df[(df["Date"] >= pd.to_datetime(start_date)) & (df["Date"] <= pd.to_datetime(end_date))]

    with st.expander("🏙️ Lọc theo thành phố"):
        cities = df["City"].unique().tolist()
        selected_cities = st.multiselect("Chọn thành phố", cities, default=cities)
        df_filtered = df_filtered[df_filtered["City"].isin(selected_cities)]

    st.write(f"📄 Số dòng dữ liệu: {len(df_filtered)}")

    selected_feature = st.selectbox("🧪 Chọn biến cần xem", df.columns[1:-4])
    fig, ax = plt.subplots()
    df_filtered.groupby("Date")[selected_feature].mean().plot(ax=ax)
    ax.set_title(f"Biểu đồ {selected_feature} theo thời gian")
    ax.set_xlabel("Date")
    ax.set_ylabel(selected_feature)
    st.pyplot(fig)

elif page == "🔮 Dự đoán PM2.5":
    st.header("🔮 Dự đoán nồng độ bụi mịn PM2.5")

    with st.form("prediction_form"):
        st.subheader("📥 Nhập các giá trị giả định:")
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

        submitted = st.form_submit_button("Dự đoán")

    if submitted:
        input_data = {
            "PM10": PM10, "NO2": NO2, "NO": NO, "NOx": NOx,
            "NH3": NH3, "CO": CO, "SO2": SO2, "O3": O3,
            "Benzene": Benzene, "Toluene": Toluene, "Xylene": Xylene,
            "Month": Month
        }
        pred_df = pd.DataFrame([input_data])

        try:
            result = model.predict(pred_df)[0]
            st.success(f"✅ Dự đoán PM2.5: **{round(result, 2)} µg/m³**")
        except Exception as e:
            st.error(f"❌ Lỗi khi dự đoán: {e}")
            st.write("📌 Kiểm tra input:")
            st.write("Số đặc trưng mô hình yêu cầu:", model.n_features_in_)
            st.write("Input shape:", pred_df.shape)
            st.write("Input columns:")
            st.write(pred_df.columns.tolist())
