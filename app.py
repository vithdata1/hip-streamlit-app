# app.py

import streamlit as st
import torch
from torchvision import transforms
from PIL import Image

# โหลดโมเดล (ResNet ในตัวอย่างนี้)
model = torch.load("best_model.pth", map_location=torch.device('cpu'))
model.eval()

# ฟังก์ชันสำหรับทำนายผลจากภาพ
def predict(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = transform(image).unsqueeze(0)  # ปรับขนาดรูปภาพให้เข้ากับโมเดล
    with torch.no_grad():
        prediction = model(image)
    return prediction.argmax().item()  # ส่งค่าทำนายที่มีความน่าจะเป็นสูงสุด

# ส่วนการสร้างอินเทอร์เฟซ Streamlit
st.title("Image Classification with ResNet")
st.write("Upload an image to classify.")

# อัปโหลดไฟล์ภาพ
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # แสดงภาพที่อัปโหลด
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # เมื่อกดปุ่ม Predict
    if st.button('Predict'):
        label = predict(image)
        st.write(f"Prediction: {label}")
