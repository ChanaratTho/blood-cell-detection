import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import os

# โหลดโมเดล
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# ตัวอย่างรูปภาพในโฟลเดอร์ samples
sample_images = {
    "ตัวอย่างที่ 1": "samples/sample1.jpg",
    "ตัวอย่างที่ 2": "samples/sample2.jpg",
    "ตัวอย่างที่ 3": "samples/sample3.jpg"
}

# Streamlit UI
st.title("Blood Cell Detection")

# เลือกระหว่างอัปโหลดรูปหรือใช้ตัวอย่าง
option = st.radio("เลือกวิธีทดสอบ:", ["อัปโหลดรูปภาพ", "ใช้รูปตัวอย่าง"])

if option == "อัปโหลดรูปภาพ":
    uploaded_file = st.file_uploader("Upload a blood cell image...", type=["jpg", "png", "jpeg"])
else:
    selected_sample = st.selectbox("เลือกรูปตัวอย่าง:", list(sample_images.keys()))
    sample_path = sample_images[selected_sample]
    image = Image.open(sample_path)
    st.image(image, caption=selected_sample, use_column_width=True)
    uploaded_file = sample_path

if uploaded_file:
    # อ่านรูปภาพ
    if option == "อัปโหลดรูปภาพ":
        image = Image.open(uploaded_file).convert("RGB")
    else:
        image = Image.open(uploaded_file)
    
    image_np = np.array(image)

    # ทำนายผลด้วย YOLO
    results = model.predict(image_np, conf=0.25)
    result = results[0]

    # วาดกล่อง Bounding Box
    plotted_img = cv2.cvtColor(result.plot(), cv2.COLOR_BGR2RGB)

    # แสดงผลลัพธ์
    st.image(plotted_img, caption="ผลลัพธ์การตรวจจับ", use_column_width=True)

    # นับจำนวนวัตถุ
    counts = {model.names[i]: 0 for i in model.names}
    for box in result.boxes:
        class_id = int(box.cls)
        counts[model.names[class_id]] += 1

    # แสดงจำนวน
    st.subheader("จำนวนที่ตรวจจับได้")
    for name, count in counts.items():
        st.markdown(f"- **{name}**: {count}")