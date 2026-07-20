import streamlit as st
import cv2
import numpy as np
import h5py
import tensorflow as tf
from PIL import Image

st.set_page_config(page_title="Nhận diện Hành vi Học sinh", page_icon="🎓", layout="wide")
st.title("🎓 HỆ THỐNG NHẬN DIỆN HÀNH VI HỌC SINH TRONG LỚP HỌC")
st.subheader("Trường THCS Thị trấn Long Thành - Luận văn Thạc sĩ CNTT")
st.markdown("---")

@st.cache_resource
def load_tm_model(model_path):
    with h5py.File(model_path, 'r+') as f:
        model_config = f.attrs.get('model_config')
        if model_config and isinstance(model_config, str):
            model_config = model_config.replace('"groups": 1,', '').replace('"groups": 1', '')
            f.attrs['model_config'] = model_config
    return tf.keras.models.load_model(model_path, compile=False)

try:
    model = load_tm_model("keras_model.h5")
    class_names = [line.strip() for line in open("labels.txt", "r", encoding="utf-8").readlines()]
    st.sidebar.success("✅ Đã nạp mô hình Học sâu thành công!")
except Exception as e:
    st.sidebar.error(f"❌ Lỗi nạp mô hình: {e}")

st.sidebar.header("⚙️ Cấu hình hệ thống")
confidence_threshold = st.sidebar.slider("Ngưỡng độ tin cậy (%)", 30, 100, 60) / 100.0

uploaded_file = st.file_uploader("Tải lên hình ảnh học sinh trong lớp...", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img_np = np.array(image)
    
    resized = cv2.resize(img_np, (224, 224), interpolation=cv2.INTER_AREA)
    img_array = np.asarray(resized, dtype=np.float32).reshape(1, 224, 224, 3)
    normalized = (img_array / 127.5) - 1.0
    
    prediction = model.predict(normalized, verbose=0)
    index = np.argmax(prediction)
    conf = prediction[0][index]
    label = class_names[index]
    
    col1, col2 = st.columns([2, 1])
    with col1:
        display_img = img_np.copy()
        text = f"{label} ({conf*100:.1f}%)"
        cv2.putText(display_img, text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        st.image(display_img, caption="Kết quả phân tích hành vi", use_column_width=True)
        
    with col2:
        st.metric(label="Hành vi phát hiện:", value=label, delta=f"{conf*100:.1f}%")
        if conf >= confidence_threshold:
            st.success(f"Kết luận: Học sinh đang **{label}**")
        else:
            st.warning("Mức độ tin cậy chưa đạt ngưỡng!")
