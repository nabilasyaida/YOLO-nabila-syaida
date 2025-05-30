from ultralytics import YOLO
import streamlit as st
from collections import Counter
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os

st.set_page_config(page_title="YOLO Acne Detection", layout="centered")
st.title("YOLO Acne Detection")

# Load model
model_path = "best.pt"
if not os.path.exists(model_path):
    st.error("Model 'best.pt' tidak ditemukan. Silakan pastikan file sudah tersedia.")
    st.stop()
else:
    model = YOLO(model_path)

# Cek gambar target
image_path = "WHITEHEADS DAN PUSTULE.jpg"
if not os.path.exists(image_path):
    st.error(f"Gambar '{image_path}' tidak ditemukan.")
    st.stop()

# Deteksi
results = model(image_path)[0]
class_ids = results.boxes.cls.cpu().numpy()
class_names = results.names
counts = Counter(class_ids)

# Gambar hasil deteksi
img_with_boxes = results.plot()
img_pil = Image.fromarray(img_with_boxes)

# Ganti background dengan gambar baru
bg_path = "Pastel Pink Holographic Gradient Mouse Pad Backgrou_
