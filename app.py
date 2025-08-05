import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title("🌫️ Phân tích chất lượng không khí (PM2.5 và các chất khác)")

# Load dữ liệu
@st.cache_data
def load_data():
    df = pd.read_csv("filtered_city_data.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    return df

data = load_data()

# Sidebar
st.sidebar.header("🎛️ Bộ lọc")

cities = st.sidebar.multiselect("Chọn thành phố", data["City"].unique(), default=data["City"].unique())
date_range = st.sidebar.date_input("Khoảng thời gian", [data["Date"].min(), data["Date"].max()])
pollutant = st.sidebar.selectbox("Chọn biến ô nhiễm", 
                                  options=[col for col in data.columns if col not in ['Date', 'City']])

# Lọc dữ liệu
filtered = data[data["City"].isin(cities)]
filtered = filtered[(filtered["Date"] >= pd.to_datetime(date_range[0])) & 
                    (filtered["Date"] <= pd.to_datetime(date_range[1]))]

# Cảnh báo nếu dữ liệu trống
if filtered.empty:
    st.warning("⚠️ Không có dữ liệu trong khoảng bạn chọn.")
    st.stop()

# Block đầu cuối của dữ liệu
st.markdown("### 🧾 Thống kê đầu và cuối của biến đã chọn")
for city in cities:
    city_data = filtered[filtered["City"] == city].sort_values("Date")
    if not city_data.empty:
        st.markdown(f"**{city}**")
        col1, col2 = st.columns(2)
        with col1:
            st.metric(label=f"🔽 Đầu khoảng: {city_data['Date'].iloc[0].date()}", 
                      value=city_data[pollutant].iloc[0])
        with col2:
            st.metric(label=f"🔼 Cuối khoảng: {city_data['Date'].iloc[-1].date()}", 
                      value=city_data[pollutant].iloc[-1])

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

# Tải dữ liệu đã lọc
st.download_button("📥 Tải dữ liệu đã lọc", data=filtered.to_csv(index=False), file_name="filtered_pm_data.csv")
