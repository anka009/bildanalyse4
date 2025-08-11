import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import os
import json

st.set_page_config(page_title="🧬 Lernfähiger Zellkern-Zähler", layout="wide")
st.title("🧬 Interaktiver Lern-Zellkern-Zähler")

# Ordner für Trainingsdaten (Korrekturen)
os.makedirs("training_data", exist_ok=True)

# --- Datei-Upload ---
uploaded_file = st.file_uploader("🔍 Bild hochladen", type=["jpg", "png", "tif", "tiff"])

if uploaded_file:
    # PIL Image laden (wichtig für st_canvas)
    image_pil = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image_pil)

    # --- Sidebar Parameter ---
    st.sidebar.header("⚙️ Parameter")
    clip_limit = st.sidebar.slider("CLAHE Kontrastverstärkung", 1.0, 5.0, 2.0, 0.1)
    min_size = st.sidebar.slider("Mindestfläche (Pixel)", 10, 10000, 500, 10)

    # --- CLAHE Vorverarbeitung ---
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # --- Otsu Thresholding ---
    otsu_thresh, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, thresh = cv2.threshold(gray, otsu_thresh, 255, cv2.THRESH_BINARY)

    # --- Invert falls Zellkerne dunkel sind ---
    if np.mean(gray[thresh == 255]) > np.mean(gray[thresh == 0]):
        thresh = cv2.bitwise_not(thresh)

    # --- Morphologische Öffnung ---
    kernel = np.ones((3, 3), np.uint8)
    clean = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    # --- Konturen finden ---
    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) >= min_size]

    centers = []
    for c in contours:
        M = cv2.moments(c)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            centers.append((cx, cy))

    # --- Markiertes Bild zum Anzeigen ---
    marked = image_np.copy()
    for (x, y) in centers:
        cv2.circle(marked, (x, y), 6, (255, 0, 0), 2)  # blau

    # --- Canvas für manuelle Korrektur ---
    st.subheader("✏️ Manuelle Korrektur: Punkte hinzufügen")

    canvas_result = st_canvas(
        fill_color="rgba(255, 0, 0, 0.5)",
        stroke_width=5,
        stroke_color="#FF0000",
        background_image=image_pil,  # PIL Image hier verwenden!
        update_streamlit=True,
        height=image_pil.height,
        width=image_pil.width,
        drawing_mode="point",
        key="canvas",
    )

    manual_points = []
    if canvas_result.json_data and "objects" in canvas_result.json_data:
        for obj in canvas_result.json_data["objects"]:
            if "left" in obj and "top" in obj:
                manual_points.append((int(obj["left"]), int(obj["top"])))

    # --- Alle Punkte kombinieren ---
    all_points = centers + manual_points

    # --- Ergebnis anzeigen ---
    marked_manual = image_np.copy()
    for (x, y) in centers:
        cv2.circle(marked_manual, (x, y), 6, (255, 0, 0), 2)  # automatisch blau
    for (x, y) in manual_points:
        cv2.circle(marked_manual, (x, y), 6, (0, 255, 0), 2)  # manuell grün

    st.image(marked_manual, caption=f"Erkannte Kerne (blau) + Manuelle Punkte (grün)", use_container_width=True)

    # --- CSV Export ---
    df = pd.DataFrame(all_points, columns=["X", "Y"])
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 Zellkerne CSV exportieren", data=csv, file_name="zellkerne.csv", mime="text/csv")

    # --- Feedback speichern ---
    if st.button("💾 Feedback speichern"):
        filename = os.path.join("training_data", f"{uploaded_file.name}_feedback.json")
        with open(filename, "w") as f:
            json.dump({"image": uploaded_file.name, "points": all_points}, f)
        st.success(f"Feedback in {filename} gespeichert!")

else:
    st.info("Bitte lade zuerst ein Bild hoch.")
