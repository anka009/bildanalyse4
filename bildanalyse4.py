import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from streamlit_drawable_canvas import st_canvas

st.set_page_config(page_title="🧬 Lernfähiger Zellkern-Zähler", layout="wide")
st.title("🧬 Interaktiver & Lernfähiger Zellkern-Zähler")

# --- Datei-Upload ---
uploaded_file = st.file_uploader("📂 Bild hochladen", type=["jpg", "png", "tif"])

if uploaded_file:
    # Bild laden
    image = np.array(Image.open(uploaded_file).convert("RGB"))

    # --- Parameter ---
    st.sidebar.header("⚙️ Parameter")
    clip_limit = st.sidebar.slider("CLAHE Kontrastverstärkung", 1.0, 5.0, 2.0, 0.1)
    min_distance = st.sidebar.slider("Min. Kernabstand (Pixel)", 5, 50, 20, 1)
    min_size = st.sidebar.slider("Mindestfläche (Pixel)", 10, 10000, 200, 10)

    # --- Vorverarbeitung ---
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    # --- Threshold ---
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # --- Morphologie ---
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    # --- Hintergrund ---
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    # --- Vordergrund ---
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

    # --- Marker für Watershed ---
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    # --- Watershed ---
    markers = cv2.watershed(image.copy(), markers)

    # --- Konturen extrahieren ---
    centers = []
    for marker_id in np.unique(markers):
        if marker_id <= 1:
            continue
        mask = np.uint8(markers == marker_id)
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            if cv2.contourArea(c) >= min_size:
                M = cv2.moments(c)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    centers.append((cx, cy))

    # --- Anzeige vorbereiten ---
    marked = image.copy()
    for (x, y) in centers:
        cv2.circle(marked, (x, y), 6, (255, 0, 0), 2)

    # --- Interaktive Korrektur ---
    st.subheader("🖱 Manuelle Korrektur")
    canvas_result = st_canvas(
        fill_color="rgba(255, 0, 0, 0.5)",
        stroke_width=3,
        background_image=Image.fromarray(marked),
        update_streamlit=True,
        height=image.shape[0],
        width=image.shape[1],
        drawing_mode="point",
        point_display_radius=6,
        key="canvas",
    )

    # --- Klicks übernehmen ---
    manual_points = []
    if canvas_result.json_data is not None:
        for obj in canvas_result.json_data["objects"]:
            px, py = int(obj["left"]), int(obj["top"])
            manual_points.append((px, py))

    # --- Alle Punkte zusammenführen ---
    all_points = centers + manual_points

    # --- CSV-Export ---
    df = pd.DataFrame(all_points, columns=["X", "Y"])
    st.write(f"**Gefundene Zellkerne (inkl. manuell hinzugefügte):** {len(all_points)}")
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("📥 CSV exportieren", data=csv, file_name="zellkerne_annotiert.csv", mime="text/csv")

    # --- Anzeige ---
    st.image(marked, caption="Automatisch erkannte Kerne (blau) + Manuelle Ergänzungen (rot)", use_container_width=True)
