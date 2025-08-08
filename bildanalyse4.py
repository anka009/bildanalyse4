import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.title("Zellkern-Zähler")

# Bild-Upload
uploaded_file = st.file_uploader("Bild hochladen", type=["jpg", "png", "tif"])
if uploaded_file:
    # Bild laden
    image = np.array(Image.open(uploaded_file).convert("RGB"))

    # Graustufen
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Optional: Kontrastverbesserung (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)

    # Schwellenwert (automatisch mit Otsu)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Invertieren (falls Kerne dunkel sind)
    thresh = cv2.bitwise_not(thresh)

    # Kleine Pixel entfernen
    kernel = np.ones((3,3), np.uint8)
    clean = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    # Zellkerne trennen (Distanz-Transform + Watershed)
    dist_transform = cv2.distanceTransform(clean, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.5*dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)

    # Konturen finden
    contours, _ = cv2.findContours(sure_fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    count = len(contours)

    # Zellkerne markieren
    marked = image.copy()
    for c in contours:
        (x, y), radius = cv2.minEnclosingCircle(c)
        cv2.circle(marked, (int(x), int(y)), int(radius), (255, 0, 0), 2)

    st.image(marked, caption=f"Gefundene Zellkerne: {count}", use_column_width=True)
    st.success(f"Anzahl Zellkerne: {count}")
