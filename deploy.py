import streamlit as st
import matplotlib.pyplot as plt
from collections import Counter
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os

st.title("YOLO Acne Detection")

# Cek apakah model tersedia
if not os.path.exists("best.pt"):
    st.error("File best.pt tidak ditemukan!")
else:
    model = YOLO("best.pt")

    # Cek gambar
    image_path = "WHITEHEADS DAN PUSTULE.jpg"
    if not os.path.exists(image_path):
        st.error(f"Gambar '{image_path}' tidak ditemukan.")
    else:
        results = model(image_path)[0]

        class_ids = results.boxes.cls.cpu().numpy()
        class_names = results.names
        counts = Counter(class_ids)

        img_with_boxes = results.plot()
        img_pil = Image.fromarray(img_with_boxes)

        bg_path = "bccbd03b-a912-417e-ae18-7918fda5d67e.jpg"
        if not os.path.exists(bg_path):
            st.warning(f"Background '{bg_path}' tidak ditemukan. Menampilkan hasil deteksi saja.")
            background = img_pil
        else:
            background = Image.open(bg_path).convert("RGB")
            background = background.resize(img_pil.size)
            background.paste(img_pil, (0, 0), img_pil if img_pil.mode == 'RGBA' else None)

        # Draw class count
        draw = ImageDraw.Draw(background)
        try:
            font = ImageFont.truetype("arial.ttf", 24)
        except:
            font = ImageFont.load_default()

        y_offset = 30
        for i, (class_id, count) in enumerate(counts.items()):
            name = class_names[int(class_id)]
            label = f"{name}: {count}"
            draw.text((10, y_offset + i * 30), label, font=font, fill=(255, 0, 0))

        # Tampilkan hasil
        st.image(background, caption="Hasil Deteksi dengan YOLOv11", use_column_width=True)
