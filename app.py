st.header("🔮 Dự đoán nồng độ bụi mịn PM2.5")

tab1, tab2 = st.tabs(["📉 Dự đoán nhanh (PM10 & NO2)", "🧪 Dự đoán chi tiết (toàn bộ đặc trưng)"])

# ==== TAB 1: DỰ ĐOÁN NHANH ====
with tab1:
    st.subheader("📉 Dự đoán từ thành phố + ngày + PM10 + NO2")
    
    col1, col2 = st.columns(2)
    with col1:
        pred_city = st.selectbox("Thành phố", df["City"].unique())
        pred_date = st.date_input("Ngày cần dự đoán", value=date(2020, 1, 1), min_value=df["Date"].min().date(), max_value=df["Date"].max().date())

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

    # Mã hóa nếu mô hình yêu cầu
    if "City" in model.feature_names_in_:
        pred_df_1["City"] = pred_df_1["City"].astype("category").cat.codes

    if st.button("🧮 Dự đoán PM2.5 (từ PM10 & NO2)"):
        try:
            result = model.predict(pred_df_1)
            st.success(f"✅ PM2.5 ước tính: **{round(result[0], 2)} µg/m³**")
        except Exception as e:
            st.error(f"❌ Lỗi khi dự đoán: {e}")
            st.write("Columns:", pred_df_1.columns.tolist())

# ==== TAB 2: DỰ ĐOÁN CHI TIẾT ====
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
