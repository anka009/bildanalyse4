import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import io
import json
import os

st.set_page_config(page_title="Interaktiver Lern-Zellkern-Zähler", layout="wide")
st.title("🧬 Interaktiver Lern-Zellkern-Zähler")

# Ordner für Trainingsdaten
os.makedirs("training_data", exist_ok=True)

# --- Datei-Upload ---
uploaded_file = st.file_uploader("🔍 Bild hochladen", type=["jpg", "png", "tif"])

if uploaded_file:
    # Bild laden
    image_pil = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image_pil)

    # --- Parameter ---
    st.sidebar.header("⚙️ Parameter")
    clip_limit = st.sidebar.slider("CLAHE Kontrastverstärkung", 1.0, 5.0, 2.0, 0.1)
    min_size = st.sidebar.slider("Mindestfläche (Pixel)", 10, 10000, 1000, 10)

    # --- CLAHE ---
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # --- Otsu Threshold ---
    otsu_thresh, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, thresh = cv2.threshold(gray, otsu_thresh, 255, cv2.THRESH_BINARY)

    # --- Invert falls nötig ---
    if np.mean(gray[thresh == 255]) > np.mean(gray[thresh == 0]):
        thresh = cv2.bitwise_not(thresh)

    # --- Morphologische Filter ---
    kernel = np.ones((3, 3), np.uint8)
    clean = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    # --- Konturen finden ---
    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) >= min_size]

    centers = []
    for c in contours:
        M = cv2.moments(c)
        if M["m00"] != 0:
            centers.append((int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])))

    # --- Canvas für manuelle Korrektur ---
    st.subheader("✏️ Manuelle Korrektur")
    canvas_result = st_canvas(
        fill_color="rgba(255, 0, 0, 0.5)",
        stroke_width=2,
        stroke_color="#ff0000",
        background_image=image_pil,  # MUSS PIL sein
        update_streamlit=True,
        height=image_np.shape[0],
        width=image_np.shape[1],
        drawing_mode="point",
        key="canvas",
    )

    # --- Punkte aus Canvas extrahieren ---
    manual_points = []
    if canvas_result.json_data is not None:
        for obj in canvas_result.json_data["objects"]:
            if "left" in obj and "top" in obj:
                manual_points.append((int(obj["left"]), int(obj["top"])))

    # --- Finale Punkte kombinieren ---
    all_points = centers + manual_points

    # --- CSV Export ---
    df = pd.DataFrame(all_points, columns=["X", "Y"])
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 CSV exportieren", data=csv, file_name="zellkerne.csv", mime="text/csv")

    # --- Bild mit Punkten anzeigen ---
    marked = image_np.copy()
    for (x, y) in all_points:
        cv2.circle(marked, (x, y), 6, (255, 0, 0), 2)

    st.image(marked, caption=f"Gefundene + manuell hinzugefügte Zellkerne: {len(all_points)}", use_container_width=True)

    # --- Lernfunktion: Punkte speichern ---
    if st.button("💾 Feedback speichern (Training)"):
        filename = os.path.join("training_data", f"{uploaded_file.name}_labels.json")
        with open(filename, "w") as f:
            json.dump({"image": uploaded_file.name, "points": all_points}, f)
        st.success(f"Feedback gespeichert in {filename}")
else:
    st.info("Bitte zuerst ein Bild hochladen.")
