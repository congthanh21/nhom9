
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# 1. CẤU HÌNH TRANG WEB & ẢNH NỀN
st.set_page_config(page_title="Dự Đoán Giá Nhà Hà Nội", layout="centered")

page_bg_img = """
<style>
[data-testid="stAppViewContainer"] {
    background-image: url("https://images.unsplash.com/photo-1565610222536-ef125c59da2e?q=80&w=2070&auto=format&fit=crop");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    background-attachment: fixed;
}
[data-testid="stSidebar"] {
    background-color: rgba(255, 255, 255, 0.9);
}
[data-testid="stHeader"], [data-testid="stToolbar"] {
    background-color: rgba(0,0,0,0);
}
.block-container {
    background-color: rgba(255, 255, 255, 0.9);
    padding: 2rem;
    border-radius: 10px;
    margin-top: 2rem;
}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

st.title("🏡 Demo Dự Đoán Giá Nhà Hà Nội")

# 2. LOAD MODEL VÀ DỮ LIỆU
@st.cache_resource
def load_resources():
    # Load model
    model_data = joblib.load('housing_model_final.pkl')
    
    # Load data gốc để lấy danh sách Quận -> Phường
    try:
        df_raw = pd.read_csv('housing_data_cleaned_v3.csv')
    except:
        # Fallback nếu tên file khác
        try:
            df_raw = pd.read_csv('housing_data_cleaned_v3.csv')
        except:
             st.error("Không tìm thấy file dữ liệu CSV!")
             st.stop()
             
    # Tạo map Quận -> Phường
    district_ward_map = df_raw.groupby('district')['ward'].unique().to_dict()
    
    return model_data, district_ward_map

try:
    data, location_map = load_resources()
    gbt = data['gbt_model']
    rf = data['rf_model']
    mappings = data['mappings']
    global_mean = data['global_mean']
    cat_cols = data['cat_cols']
    model_features = data['features']
except FileNotFoundError:
    st.error("Thiếu file model hoặc dữ liệu!")
    st.stop()

# 3. SIDEBAR NHẬP LIỆU
st.sidebar.header("Thông tin ngôi nhà")

# Chọn Quận
districts = sorted(list(location_map.keys()))
selected_district = st.sidebar.selectbox("Quận/Huyện", districts)

# Chọn Phường (Lọc theo Quận)
valid_wards = sorted(list(location_map[selected_district]))
selected_ward = st.sidebar.selectbox("Phường/Xã", valid_wards)

# Chọn Loại nhà (type_of_housing) - GIỮ NGUYÊN
types = sorted(list(mappings['type_of_housing'].keys()))
type_housing = st.sidebar.selectbox("Loại nhà", types)

# Chọn Giấy tờ
papers = sorted(list(mappings['legal_paper'].keys()))
legal = st.sidebar.selectbox("Giấy tờ pháp lý", papers)

# Các thông số số học
area = st.sidebar.number_input("Diện tích (m2)", min_value=10.0, value=50.0, step=1.0)
floors = st.sidebar.number_input("Số tầng", min_value=1, value=3, step=1)

# 4. XỬ LÝ DỮ LIỆU
def preprocess_input():
    input_data = {
        'district': selected_district, 
        'ward': selected_ward, 
        'street': 'Unknown', # Mặc định là Unknown vì không nhập
        'type_of_housing': type_housing, 
        'legal_paper': legal,
        'num_floors': floors, 
        'num_bed_rooms': 3, # Mặc định 3 phòng ngủ
        'squared_meter_area': area
    }
    
    df_input = pd.DataFrame([input_data])
    
    # Target Encoding
    for col in cat_cols:
        mapping = mappings[col]
        # Nếu giá trị không có trong map, dùng global_mean
        encoded_val = mapping.get(df_input[col][0], global_mean)
        df_input[col + '_encoded'] = encoded_val
    
    return df_input[model_features]

# 5. DỰ ĐOÁN & HIỂN THỊ KẾT QUẢ
if st.button("🔮 Dự đoán giá"):
    try:
        X_new = preprocess_input()
        
        # Dự đoán Log Price từ 2 model
        pred_log_rf = rf.predict(X_new)[0]
        pred_log_gbt = gbt.predict(X_new)[0]
        
        # Ensemble: 60% RF + 40% GBT
        weighted_log_pred = (pred_log_rf * 0.6) + (pred_log_gbt * 0.4)
        
        # Chuyển đổi Log về giá thực
        final_price = np.expm1(weighted_log_pred) 
        price_per_m2 = final_price / area

        # Hiển thị kết quả
        st.success(f"💰 Giá dự đoán: **{final_price/1000:,.2f} Tỷ VNĐ**")
        st.info(f"Đơn giá trung bình: {price_per_m2:,.1f} Triệu/m2")
            
    except Exception as e:
        st.error(f"Lỗi khi dự đoán: {e}")
