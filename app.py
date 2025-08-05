# ======================== TAB 2: DỰ ĐOÁN ========================
with tabs[1]:
    st.subheader("🔮 Dự đoán nồng độ PM2.5")

    pred_date = st.date_input("📅 Chọn ngày dự đoán", value=date(2020, 1, 1), 
                              min_value=data["Date"].min().date(), 
                              max_value=data["Date"].max().date())
    city = st.selectbox("🏩 Chọn thành phố", data["City"].unique())
    pm10_value = st.number_input("🔸 PM10", min_value=0.0, value=100.0)

    # Encode tên thành phố thành số
    city_mapping = {c: idx for idx, c in enumerate(data["City"].unique())}

    # Trung bình các cột số để điền vào những biến còn lại
    avg_dict = data.select_dtypes(include='number').mean().to_dict()

    # Tạo dict đầu vào (chỉ chứa các feature mà model cần)
    input_dict = {
        "City": city_mapping.get(city, 0),
        "Day": pred_date.day,
        "Month": pred_date.month,
        "PM10": pm10_value,
        "NO": avg_dict.get("NO", 0),
        "NO2": avg_dict.get("NO2", 0),
        "NOx": avg_dict.get("NOx", 0),
        "NH3": avg_dict.get("NH3", 0),
        "CO": avg_dict.get("CO", 0),
        "SO2": avg_dict.get("SO2", 0),
        "O3": avg_dict.get("O3", 0),
        "Benzene": avg_dict.get("Benzene", 0),
        # ❌ KHÔNG thêm 'Toluene' hoặc 'Xylene' nếu mô hình không cần
    }

    input_df = pd.DataFrame([input_dict])

    # Nếu model hỗ trợ, chỉ giữ đúng cột mà model đã học
    try:
        input_df = input_df[model.feature_names_in_]
    except:
        pass  # nếu không có feature_names_in_, dùng nguyên input_dict ở trên

    st.markdown("📋 Dữ liệu đưa vào mô hình:")
    st.dataframe(input_df)

    if st.button("🧲 Dự đoán PM2.5"):
        try:
            result = model.predict(input_df)
            st.success(f"✅ Dự đoán PM2.5: **{round(float(result[0]), 2)} µg/m³**")
        except Exception as e:
            st.error(f"❌ Lỗi khi dự đoán: {e}")
